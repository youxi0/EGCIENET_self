#!/usr/bin/env python3

"""
V10 实验版 ONNX 图改写脚本。

以已经验证过的 V7 Q/DQ 模型为基线，将三个 block1 中的下列链路：

    Transpose -> Reshape -> DWConv -> Reshape -> Transpose -> exact GELU

替换为一个 TensorRT IPluginV3 节点。插件输入和输出都保持 token-major
[B, H*W, C] 布局，因此不再真正生成 NCHW 中间张量。

本脚本必须在 ModelOpt 之后执行。不要对含自定义插件节点的结果再次运行
ModelOpt，否则可能破坏已经验证过的 V7 Q/DQ 边界。
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import onnx
from onnx import ModelProto, NodeProto, helper


PLUGIN_OP = "EGCINET_Block1FusedDwconvGelu"
PLUGIN_DOMAIN = "egcinet"
# REGISTER_TENSORRT_PLUGIN 会把 Creator 静态注册到 TensorRT 默认命名空间；
# ONNX 自定义算子仍使用独立的 egcinet domain，避免被标准 ONNX 算子误识别。
PLUGIN_NAMESPACE = ""
PLUGIN_VERSION = "1"
BLOCK_INDICES = (0, 1, 2)


def build_maps(
    model: ModelProto,
) -> Tuple[Dict[str, NodeProto], Dict[str, NodeProto], Dict[str, List[NodeProto]]]:
    """一次遍历建立节点名称、tensor 生产者和 tensor 消费者三张索引表。"""
    nodes_by_name: Dict[str, NodeProto] = {}
    producer: Dict[str, NodeProto] = {}
    consumers: Dict[str, List[NodeProto]] = {}
    for node in model.graph.node:
        if node.name:
            if node.name in nodes_by_name:
                raise RuntimeError("duplicate node name: {}".format(node.name))
            nodes_by_name[node.name] = node
        for output_name in node.output:
            if not output_name:
                continue
            if output_name in producer:
                raise RuntimeError("duplicate tensor producer: {}".format(output_name))
            producer[output_name] = node
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    return nodes_by_name, producer, consumers


def require_node(
    nodes_by_name: Dict[str, NodeProto],
    name: str,
    op_type: str,
    prefix: str,
) -> NodeProto:
    """按完整名称和 op_type 精确定位节点；不允许模糊匹配后继续改图。"""
    node = nodes_by_name.get(name)
    if node is None:
        candidates = sorted(
            candidate_name
            for candidate_name in nodes_by_name
            if candidate_name.startswith(prefix)
        )
        raise RuntimeError(
            "missing node {}. Nodes under the same block: {}".format(
                name,
                candidates,
            )
        )
    if node.op_type != op_type:
        raise RuntimeError(
            "{} must be {}, got {}".format(name, op_type, node.op_type)
        )
    return node


def require_direct_edge(
    source: NodeProto,
    target: NodeProto,
    consumers: Dict[str, List[NodeProto]],
    exclusive: bool = True,
    allowed_extra_consumers: Sequence[Tuple[str, str]] = (),
) -> None:
    """检查两个节点是否直连，并可按节点名和类型放行指定的额外消费者。"""
    if len(source.output) != 1 or source.output[0] not in target.input:
        raise RuntimeError(
            "expected direct edge {} -> {}".format(source.name, target.name)
        )
    if exclusive:
        actual = consumers.get(source.output[0], [])
        target_consumers = [node for node in actual if node.name == target.name]
        allowed_extra = set(allowed_extra_consumers)
        unexpected = [
            node
            for node in actual
            if node.name != target.name
            and (node.name, node.op_type) not in allowed_extra
        ]
        if len(target_consumers) != 1 or unexpected:
            raise RuntimeError(
                "{} output has unexpected consumers: {}".format(
                    source.name,
                    [node.name for node in actual],
                )
            )


def trace_optional_casts(
    start: NodeProto,
    target: NodeProto,
    consumers: Dict[str, List[NodeProto]],
) -> List[NodeProto]:
    """沿单消费者链追踪零个或多个 Cast，用于兼容 Erf 前后的精度转换。"""
    if len(start.output) != 1:
        raise RuntimeError("{} must have one output".format(start.name))

    casts: List[NodeProto] = []
    current_tensor = start.output[0]
    while current_tensor not in target.input:
        next_nodes = consumers.get(current_tensor, [])
        if len(next_nodes) != 1 or next_nodes[0].op_type != "Cast":
            raise RuntimeError(
                "expected optional Cast chain {} -> {}, got consumers {}".format(
                    start.name,
                    target.name,
                    [(node.name, node.op_type) for node in next_nodes],
                )
            )
        cast = next_nodes[0]
        if len(cast.output) != 1:
            raise RuntimeError("{} must have one output".format(cast.name))
        casts.append(cast)
        current_tensor = cast.output[0]

    final_consumers = consumers.get(current_tensor, [])
    if len(final_consumers) != 1 or final_consumers[0].name != target.name:
        raise RuntimeError(
            "tensor before {} has unexpected consumers: {}".format(
                target.name,
                [node.name for node in final_consumers],
            )
        )
    return casts


def require_consumer_names(
    tensor_name: str,
    expected_names: Iterable[str],
    consumers: Dict[str, List[NodeProto]],
) -> None:
    """严格核对一个 tensor 的全部消费者，防止误删仍被其他分支使用的节点。"""
    actual = sorted(node.name for node in consumers.get(tensor_name, []))
    expected = sorted(expected_names)
    if actual != expected:
        raise RuntimeError(
            "tensor {} consumers mismatch: expected {}, got {}".format(
                tensor_name,
                expected,
                actual,
            )
        )


def node_attributes(node: NodeProto) -> Dict[str, object]:
    """把 ONNX AttributeProto 转成便于校验的名称到值映射。"""
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def replace_one_block(
    model: ModelProto,
    block_index: int,
    height: int,
    width: int,
) -> None:
    """验证并替换一个 block1 的 DWConv 布局链和精确 GELU 子图。"""
    # 步骤 1：重建当前图索引。前一个 block 替换后，节点集合已经发生变化。
    nodes_by_name, producer, consumers = build_maps(model)
    prefix = "/model/block1.{}/mlp".format(block_index)
    dwconv_prefix = prefix + "/dwconv"
    act_prefix = prefix + "/act"

    # 步骤 2：按导出时的完整名称定位布局变换、DWConv 和 GELU 主链节点。
    transpose_in = require_node(
        nodes_by_name,
        dwconv_prefix + "/Transpose",
        "Transpose",
        prefix,
    )
    reshape_in = require_node(
        nodes_by_name,
        dwconv_prefix + "/Reshape",
        "Reshape",
        prefix,
    )
    conv = require_node(
        nodes_by_name,
        dwconv_prefix + "/dwconv/Conv",
        "Conv",
        prefix,
    )
    reshape_out = require_node(
        nodes_by_name,
        dwconv_prefix + "/Reshape_1",
        "Reshape",
        prefix,
    )
    transpose_out = require_node(
        nodes_by_name,
        dwconv_prefix + "/Transpose_1",
        "Transpose",
        prefix,
    )
    div = require_node(nodes_by_name, act_prefix + "/Div", "Div", prefix)
    erf = require_node(nodes_by_name, act_prefix + "/Erf", "Erf", prefix)
    add = require_node(nodes_by_name, act_prefix + "/Add", "Add", prefix)
    mul = require_node(nodes_by_name, act_prefix + "/Mul", "Mul", prefix)
    mul_final = require_node(
        nodes_by_name,
        act_prefix + "/Mul_1",
        "Mul",
        prefix,
    )

    if len(transpose_in.input) != 1:
        raise RuntimeError("{} must have one input".format(transpose_in.name))
    if len(conv.input) < 3:
        raise RuntimeError("{} must have activation, weight and bias".format(conv.name))
    if len(mul_final.output) != 1:
        raise RuntimeError("{} must have one output".format(mul_final.name))

    # 步骤 3：验证两次 Transpose 都是 token-major 与 NCHW 之间的 [0,2,1] 交换。
    transpose_in_attributes = node_attributes(transpose_in)
    transpose_out_attributes = node_attributes(transpose_out)
    if list(transpose_in_attributes.get("perm", [])) != [0, 2, 1] or \
       list(transpose_out_attributes.get("perm", [])) != [0, 2, 1]:
        raise RuntimeError(
            "block1.{} DWConv transposes must both use perm [0,2,1]".format(
                block_index
            )
        )

    # 步骤 4：验证 Conv 确实是 stride=1、pad=1 的 3x3 depthwise 卷积。
    conv_attributes = node_attributes(conv)
    if (
        int(conv_attributes.get("group", 1)) <= 1
        or list(conv_attributes.get("kernel_shape", [3, 3])) != [3, 3]
        or list(conv_attributes.get("strides", [1, 1])) != [1, 1]
        or list(conv_attributes.get("dilations", [1, 1])) != [1, 1]
        or list(conv_attributes.get("pads", [0, 0, 0, 0])) != [1, 1, 1, 1]
    ):
        raise RuntimeError(
            "{} is not a 3x3 stride-1 pad-1 depthwise Conv".format(conv.name)
        )

    # 步骤 5：向上追踪权重 Cast，明确拒绝从 V8 的量化 DWConv 权重生成 V10。
    weight_parent = producer.get(conv.input[1])
    while weight_parent is not None and weight_parent.op_type == "Cast":
        weight_parent = producer.get(weight_parent.input[0])
    if weight_parent is not None and weight_parent.op_type == "DequantizeLinear":
        raise RuntimeError(
            "{} weight is quantized; V10 must be generated from V7, not V8".format(
                conv.name
            )
        )

    require_direct_edge(transpose_in, reshape_in, consumers)
    require_direct_edge(reshape_in, conv, consumers)
    # 动态导出的 Reshape 会通过 Shape 读取 Conv 输出尺寸，再构造 Reshape_1
    # 的第二个输入。Shape 只参与形状计算，不读取实际特征值；替换完成后该支路
    # 会失去用途，并由后面的反向活性分析整体删除。
    require_direct_edge(
        conv,
        reshape_out,
        consumers,
        allowed_extra_consumers=((dwconv_prefix + "/Shape", "Shape"),),
    )
    require_direct_edge(reshape_out, transpose_out, consumers)

    # 步骤 6：验证精确 GELU：x * (1 + erf(x / sqrt(2))) * 0.5。
    # ModelOpt 可能在 Erf 前后插入 FP16->FP32 和 FP32->FP16 Cast，因此单独追踪。
    require_consumer_names(
        transpose_out.output[0],
        (div.name, mul.name),
        consumers,
    )
    casts_before_erf = trace_optional_casts(div, erf, consumers)
    casts_after_erf = trace_optional_casts(erf, add, consumers)
    require_direct_edge(add, mul, consumers)
    if transpose_out.output[0] not in mul.input:
        raise RuntimeError("{} must also consume GELU x".format(mul.name))
    require_direct_edge(mul, mul_final, consumers)

    # 步骤 7：插件直接读取 fc1 的 token-major 输出和原 DWConv 权重/bias，
    # 并复用最终 GELU tensor 名称，使下游 fc2 无需修改。
    activation = transpose_in.input[0]
    weight = conv.input[1]
    bias = conv.input[2]
    output = mul_final.output[0]

    chain: List[NodeProto] = [
        transpose_in,
        reshape_in,
        conv,
        reshape_out,
        transpose_out,
        div,
        *casts_before_erf,
        erf,
        *casts_after_erf,
        add,
        mul,
        mul_final,
    ]
    remove_names = {node.name for node in chain}
    original_nodes = list(model.graph.node)
    # 步骤 8：插件插在原 Conv 位置，而不是第一个 Transpose 位置。
    # ModelOpt 可能在 Conv 前插入产生 FP16 权重的 Cast；这样可保证插件的全部输入
    # 生产者在拓扑顺序上仍位于插件之前。
    insert_at = next(
        index for index, node in enumerate(original_nodes) if node.name == conv.name
    )

    plugin = helper.make_node(
        PLUGIN_OP,
        inputs=[activation, weight, bias],
        outputs=[output],
        name="/model/block1.{}/mlp/Block1FusedDwconvGelu".format(block_index),
        domain=PLUGIN_DOMAIN,
        height=height,
        width=width,
        plugin_namespace=PLUGIN_NAMESPACE,
        plugin_version=PLUGIN_VERSION,
    )

    # 步骤 9：在保持其他节点原始顺序的前提下，插入插件并移除已验证的旧链。
    rewritten: List[NodeProto] = []
    inserted = False
    for index, node in enumerate(original_nodes):
        if index == insert_at:
            rewritten.append(plugin)
            inserted = True
        if node.name not in remove_names:
            rewritten.append(node)
    if not inserted:
        raise RuntimeError("failed to insert plugin for block1.{}".format(block_index))

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    print(
        "[REPLACE] block1.{}: {} nodes -> 1 plugin; input={} output={}".format(
            block_index,
            len(chain),
            activation,
            output,
        )
    )


def remove_dead_nodes_and_initializers(model: ModelProto) -> None:
    """从图输出反向保留有效计算，删除旧布局链遗留的形状支路和参数。"""
    original_nodes = list(model.graph.node)
    producer_index: Dict[str, int] = {}
    for index, node in enumerate(original_nodes):
        for output_name in node.output:
            if output_name:
                producer_index[output_name] = index

    # 步骤 1：从全部图输出反向遍历生产者。只有真正参与最终输出计算的节点
    # 才会被标记为有效，因此 Conv -> Shape -> ... -> Reshape_1 的动态形状
    # 支路会随着 Reshape_1 一起被安全删除。
    live_node_indices = set()
    pending_tensors = [value.name for value in model.graph.output]
    while pending_tensors:
        tensor_name = pending_tensors.pop()
        node_index = producer_index.get(tensor_name)
        if node_index is None or node_index in live_node_indices:
            continue
        live_node_indices.add(node_index)
        pending_tensors.extend(
            input_name
            for input_name in original_nodes[node_index].input
            if input_name
        )

    kept_nodes = [
        node
        for index, node in enumerate(original_nodes)
        if index in live_node_indices
    ]
    removed_nodes = len(original_nodes) - len(kept_nodes)
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    if removed_nodes:
        print("[CLEAN  ] removed {} dead nodes".format(removed_nodes))

    # 步骤 2：按有效节点的输入集合清理已经不再使用的 shape initializer。
    graph_outputs = {value.name for value in model.graph.output}
    used_inputs = {
        input_name
        for node in model.graph.node
        for input_name in node.input
        if input_name
    }
    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in used_inputs or initializer.name in graph_outputs
    ]
    removed = len(model.graph.initializer) - len(kept_initializers)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    if removed:
        print("[CLEAN  ] removed {} unused initializers".format(removed))


def ensure_plugin_opset(model: ModelProto) -> None:
    """确保模型声明 egcinet 自定义 domain，供 ONNX checker 和 parser 识别。"""
    for opset in model.opset_import:
        if opset.domain == PLUGIN_DOMAIN:
            if opset.version != 1:
                raise RuntimeError(
                    "{} domain already uses unsupported opset {}".format(
                        PLUGIN_DOMAIN,
                        opset.version,
                    )
                )
            return
    model.opset_import.append(helper.make_operatorsetid(PLUGIN_DOMAIN, 1))


def validate_rewritten_model(model: ModelProto) -> None:
    """保存前后均执行的终检：插件数量、旧节点残留和 ONNX 合法性。"""
    # 步骤 1：必须恰好存在 block1.0、1、2 对应的三个插件节点。
    plugin_names = {
        node.name
        for node in model.graph.node
        if node.op_type == PLUGIN_OP and node.domain == PLUGIN_DOMAIN
    }
    expected = {
        "/model/block1.{}/mlp/Block1FusedDwconvGelu".format(index)
        for index in BLOCK_INDICES
    }
    if plugin_names != expected:
        raise RuntimeError(
            "plugin nodes mismatch: expected {}, got {}".format(
                sorted(expected),
                sorted(plugin_names),
            )
        )

    # 步骤 2：原布局链和 GELU 主链节点不得残留。
    old_names = set()
    for index in BLOCK_INDICES:
        prefix = "/model/block1.{}/mlp".format(index)
        old_names.update(
            {
                prefix + "/dwconv/Transpose",
                prefix + "/dwconv/Reshape",
                prefix + "/dwconv/dwconv/Conv",
                prefix + "/dwconv/Reshape_1",
                prefix + "/dwconv/Transpose_1",
                prefix + "/act/Div",
                prefix + "/act/Erf",
                prefix + "/act/Add",
                prefix + "/act/Mul",
                prefix + "/act/Mul_1",
            }
        )
    old_nodes = [node.name for node in model.graph.node if node.name in old_names]
    if old_nodes:
        raise RuntimeError("old block1 DWConv layout nodes remain: {}".format(old_nodes))

    # 步骤 3：最后交给 ONNX checker 检查拓扑、类型信息和自定义 domain 声明。
    onnx.checker.check_model(model)
    print("[PASS   ] 3/3 block1 layout chains replaced by TensorRT plugin nodes")


def parse_args() -> argparse.Namespace:
    """解析 V7 输入、V10 输出和固定 block1 空间尺寸。"""
    parser = argparse.ArgumentParser(
        description="Replace block1 DWConv layout chains in a V7 Q/DQ ONNX model",
    )
    parser.add_argument(
        "--input",
        default="deploy/egcienet_352_multiclass_qdq_v7_block1_fc1.onnx",
        help="validated V7 Q/DQ ONNX model",
    )
    parser.add_argument(
        "--output",
        default="deploy/egcienet_352_multiclass_qdq_v10_block1_fused.onnx",
        help="V10 ONNX model containing the custom TensorRT nodes",
    )
    parser.add_argument("--height", type=int, default=88)
    parser.add_argument("--width", type=int, default=88)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output model",
    )
    return parser.parse_args()


def main() -> None:
    # 步骤 1：先检查路径和参数，默认禁止覆盖已有输出。
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")
    if not input_path.is_file():
        raise FileNotFoundError("input ONNX not found: {}".format(input_path))
    if input_path.resolve() == output_path.resolve():
        raise RuntimeError("output must not overwrite the input ONNX")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "output already exists: {}; pass --force to overwrite".format(output_path)
        )

    print("[INPUT  ] {}".format(input_path))
    print("[OUTPUT ] {}".format(output_path))
    print("[SHAPE  ] block1 H={} W={}".format(args.height, args.width))
    print("[BASE   ] V7 Q/DQ; block1 DWConv remains high precision")

    # 步骤 2：读取并预检 V7，然后依次替换三个 block1。
    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    for block_index in BLOCK_INDICES:
        replace_one_block(model, block_index, args.height, args.width)

    # 步骤 3：补充自定义 opset、清理死节点并执行保存前终检。
    ensure_plugin_opset(model)
    remove_dead_nodes_and_initializers(model)
    validate_rewritten_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    # 步骤 4：重新读取磁盘文件再校验一次，防止序列化阶段破坏图结构。
    saved_model = onnx.load(str(output_path))
    validate_rewritten_model(saved_model)
    print("[DONE   ] {}".format(output_path))


if __name__ == "__main__":
    main()
