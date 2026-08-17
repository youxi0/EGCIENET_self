#!/usr/bin/env python3

"""
V11 实验版 ONNX 图改写脚本。

以已经验证过的 V7 Q/DQ 模型为基线，将三个 block1 中的：

    Transpose -> Reshape -> DWConv -> Reshape -> Transpose

替换为一个单输入 TensorRT IPluginV3 节点。原 GELU 子图完整保留，继续交给
TensorRT 做精度选择、融合和 tactic 规划。

DWConv 静态权重在本脚本中提前转换为 FP16，并按 [kernel=9][channelPair=C/2]
打包；bias 按 [channelPair=C/2] 打包。每个 ONNX int32 属性承载一个 half2 的
原始 32 位位模式，Plugin 创建时只需一次性上传，不在 enqueue 中重排权重。

本脚本必须在 ModelOpt 之后执行，不要对生成的自定义插件 ONNX 再运行 ModelOpt。
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper


PLUGIN_OP = "EGCINET_Block1PackedDwconv"
PLUGIN_DOMAIN = "egcinet"
PLUGIN_NAMESPACE = ""
PLUGIN_VERSION = "1"
BLOCK_INDICES = (0, 1, 2)


def build_maps(
    model: ModelProto,
) -> Tuple[Dict[str, NodeProto], Dict[str, NodeProto], Dict[str, List[NodeProto]]]:
    """建立节点名称、tensor 生产者和 tensor 消费者三张索引表。"""
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
    """按完整名称和 op_type 精确定位节点。"""
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
    allowed_extra_consumers: Sequence[Tuple[str, str]] = (),
) -> None:
    """检查两个节点是否直连，并精确限制源 tensor 的额外消费者。"""
    if len(source.output) != 1 or source.output[0] not in target.input:
        raise RuntimeError(
            "expected direct edge {} -> {}".format(source.name, target.name)
        )

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
                [(node.name, node.op_type) for node in actual],
            )
        )


def require_consumer_names(
    tensor_name: str,
    expected_names: Iterable[str],
    consumers: Dict[str, List[NodeProto]],
) -> None:
    """严格核对一个 tensor 的全部消费者。"""
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
    """把 ONNX AttributeProto 转换成名称到值的映射。"""
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def trace_optional_casts(
    start: NodeProto,
    target: NodeProto,
    consumers: Dict[str, List[NodeProto]],
) -> List[NodeProto]:
    """沿单消费者链追踪零个或多个 Cast。"""
    if len(start.output) != 1:
        raise RuntimeError("{} must have one output".format(start.name))

    casts: List[NodeProto] = []
    current_tensor = start.output[0]
    while current_tensor not in target.input:
        next_nodes = consumers.get(current_tensor, [])
        if len(next_nodes) != 1 or next_nodes[0].op_type != "Cast":
            raise RuntimeError(
                "expected optional Cast chain {} -> {}, got {}".format(
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


def resolve_static_tensor(
    tensor_name: str,
    producer: Dict[str, NodeProto],
    initializers: Dict[str, TensorProto],
    resolving: Set[str] = None,
) -> np.ndarray:
    """解析 initializer、Constant、Identity 和 Cast 组成的静态 tensor 链。"""
    if tensor_name in initializers:
        return np.asarray(numpy_helper.to_array(initializers[tensor_name]))

    if resolving is None:
        resolving = set()
    if tensor_name in resolving:
        raise RuntimeError("cycle while resolving static tensor {}".format(tensor_name))
    resolving.add(tensor_name)

    node = producer.get(tensor_name)
    if node is None:
        raise RuntimeError("tensor is not static: {}".format(tensor_name))

    try:
        if node.op_type == "Constant":
            attributes = node_attributes(node)
            value = attributes.get("value")
            if not isinstance(value, TensorProto):
                raise RuntimeError(
                    "{} Constant does not contain a tensor value".format(node.name)
                )
            return np.asarray(numpy_helper.to_array(value))

        if node.op_type == "Identity":
            if len(node.input) != 1:
                raise RuntimeError("{} Identity must have one input".format(node.name))
            return resolve_static_tensor(
                node.input[0],
                producer,
                initializers,
                resolving,
            )

        if node.op_type == "Cast":
            if len(node.input) != 1:
                raise RuntimeError("{} Cast must have one input".format(node.name))
            source = resolve_static_tensor(
                node.input[0],
                producer,
                initializers,
                resolving,
            )
            destination_type = int(node_attributes(node).get("to", -1))
            dtype_by_onnx_type = {
                TensorProto.FLOAT16: np.float16,
                TensorProto.FLOAT: np.float32,
                TensorProto.DOUBLE: np.float64,
            }
            dtype = dtype_by_onnx_type.get(destination_type)
            if dtype is None:
                raise RuntimeError(
                    "{} uses unsupported static Cast type {}".format(
                        node.name,
                        destination_type,
                    )
                )
            return source.astype(dtype)

        if node.op_type == "DequantizeLinear":
            raise RuntimeError(
                "{} is quantized; V11 must be generated from V7, not V8".format(
                    tensor_name
                )
            )

        raise RuntimeError(
            "unsupported static producer {} ({}) for {}".format(
                node.name,
                node.op_type,
                tensor_name,
            )
        )
    finally:
        resolving.remove(tensor_name)


def half_pairs_to_signed_words(values: np.ndarray) -> List[int]:
    """把最后一维为 2 的 FP16 数组按 half2 位模式打包成有符号 int32。"""
    pairs = np.ascontiguousarray(values, dtype=np.float16)
    if pairs.ndim < 1 or pairs.shape[-1] != 2:
        raise ValueError("half2 packing requires the last dimension to be 2")

    bits = pairs.view(np.uint16)
    words = (
        bits[..., 0].astype(np.uint32)
        | (bits[..., 1].astype(np.uint32) << np.uint32(16))
    )
    # ONNX INTS 使用有符号整数；view 保持原始 32 位位模式不变。
    return np.ascontiguousarray(words).view(np.int32).reshape(-1).tolist()


def pack_dwconv_parameters(
    weight: np.ndarray,
    bias: np.ndarray,
    node_name: str,
) -> Tuple[int, List[int], List[int]]:
    """把 [C,1,3,3] 权重转换成 [9][C/2] half2，并同时打包 bias。"""
    weight = np.asarray(weight)
    bias = np.asarray(bias)
    if weight.ndim != 4 or tuple(weight.shape[1:]) != (1, 3, 3):
        raise RuntimeError(
            "{} weight must be [C,1,3,3], got {}".format(
                node_name,
                tuple(weight.shape),
            )
        )

    channels = int(weight.shape[0])
    if channels <= 0 or (channels & 1) != 0:
        raise RuntimeError(
            "{} requires a positive even channel count, got {}".format(
                node_name,
                channels,
            )
        )
    if bias.size != channels:
        raise RuntimeError(
            "{} bias must contain {} values, got {}".format(
                node_name,
                channels,
                bias.size,
            )
        )

    # 原布局 [C,9] 先按相邻通道组成 pair，再交换成 [9,C/2,2]。
    weight_fp16 = np.ascontiguousarray(weight, dtype=np.float16).reshape(channels, 9)
    weight_pairs = weight_fp16.reshape(channels // 2, 2, 9).transpose(2, 0, 1)
    bias_pairs = np.ascontiguousarray(bias, dtype=np.float16).reshape(channels // 2, 2)

    packed_weights = half_pairs_to_signed_words(weight_pairs)
    packed_bias = half_pairs_to_signed_words(bias_pairs)
    if len(packed_weights) != 9 * channels // 2 or len(packed_bias) != channels // 2:
        raise RuntimeError("internal half2 packing length mismatch")
    return channels, packed_weights, packed_bias


def validate_exact_gelu(
    nodes_by_name: Dict[str, NodeProto],
    consumers: Dict[str, List[NodeProto]],
    prefix: str,
    gelu_input_tensor: str,
) -> None:
    """确认插件输出后仍连接原来的精确 GELU 子图。"""
    act_prefix = prefix + "/act"
    div = require_node(nodes_by_name, act_prefix + "/Div", "Div", prefix)
    erf = require_node(nodes_by_name, act_prefix + "/Erf", "Erf", prefix)
    add = require_node(nodes_by_name, act_prefix + "/Add", "Add", prefix)
    mul = require_node(nodes_by_name, act_prefix + "/Mul", "Mul", prefix)
    mul_final = require_node(nodes_by_name, act_prefix + "/Mul_1", "Mul", prefix)

    require_consumer_names(gelu_input_tensor, (div.name, mul.name), consumers)
    trace_optional_casts(div, erf, consumers)
    trace_optional_casts(erf, add, consumers)
    require_direct_edge(add, mul, consumers)
    if gelu_input_tensor not in mul.input:
        raise RuntimeError("{} must also consume GELU x".format(mul.name))
    require_direct_edge(mul, mul_final, consumers)


def replace_one_block(
    model: ModelProto,
    block_index: int,
    height: int,
    width: int,
) -> None:
    """替换一个 block1 布局链和 DWConv，同时保留原 GELU。"""
    nodes_by_name, producer, consumers = build_maps(model)
    initializers = {
        initializer.name: initializer
        for initializer in model.graph.initializer
    }
    prefix = "/model/block1.{}/mlp".format(block_index)
    dwconv_prefix = prefix + "/dwconv"

    # 步骤 1：精确定位需要删除的五个节点。
    transpose_in = require_node(
        nodes_by_name, dwconv_prefix + "/Transpose", "Transpose", prefix
    )
    reshape_in = require_node(
        nodes_by_name, dwconv_prefix + "/Reshape", "Reshape", prefix
    )
    conv = require_node(
        nodes_by_name, dwconv_prefix + "/dwconv/Conv", "Conv", prefix
    )
    reshape_out = require_node(
        nodes_by_name, dwconv_prefix + "/Reshape_1", "Reshape", prefix
    )
    transpose_out = require_node(
        nodes_by_name, dwconv_prefix + "/Transpose_1", "Transpose", prefix
    )

    if len(transpose_in.input) != 1:
        raise RuntimeError("{} must have one input".format(transpose_in.name))
    if len(conv.input) < 3:
        raise RuntimeError("{} must have activation, weight and bias".format(conv.name))
    if len(transpose_out.output) != 1:
        raise RuntimeError("{} must have one output".format(transpose_out.name))

    # 步骤 2：核对布局变换和 depthwise 卷积属性。
    if list(node_attributes(transpose_in).get("perm", [])) != [0, 2, 1] or \
       list(node_attributes(transpose_out).get("perm", [])) != [0, 2, 1]:
        raise RuntimeError(
            "block1.{} DWConv transposes must use [0,2,1]".format(block_index)
        )

    conv_attributes = node_attributes(conv)
    group = int(conv_attributes.get("group", 1))
    if (
        group <= 1
        or list(conv_attributes.get("kernel_shape", [3, 3])) != [3, 3]
        or list(conv_attributes.get("strides", [1, 1])) != [1, 1]
        or list(conv_attributes.get("dilations", [1, 1])) != [1, 1]
        or list(conv_attributes.get("pads", [0, 0, 0, 0])) != [1, 1, 1, 1]
    ):
        raise RuntimeError(
            "{} is not a 3x3 stride-1 pad-1 depthwise Conv".format(conv.name)
        )

    require_direct_edge(transpose_in, reshape_in, consumers)
    require_direct_edge(reshape_in, conv, consumers)
    require_direct_edge(
        conv,
        reshape_out,
        consumers,
        allowed_extra_consumers=((dwconv_prefix + "/Shape", "Shape"),),
    )
    require_direct_edge(reshape_out, transpose_out, consumers)

    # 步骤 3：解析实际静态参数并在 engine 构建前完成 half2 物理打包。
    weight = resolve_static_tensor(conv.input[1], producer, initializers)
    bias = resolve_static_tensor(conv.input[2], producer, initializers)
    channels, packed_weights, packed_bias = pack_dwconv_parameters(
        weight,
        bias,
        conv.name,
    )
    if group != channels:
        raise RuntimeError(
            "{} group {} does not match channels {}".format(
                conv.name,
                group,
                channels,
            )
        )

    # 步骤 4：插件复用原 Transpose_1 输出 tensor 名称，原 GELU 无需改线。
    activation = transpose_in.input[0]
    output = transpose_out.output[0]
    validate_exact_gelu(nodes_by_name, consumers, prefix, output)

    plugin = helper.make_node(
        PLUGIN_OP,
        inputs=[activation],
        outputs=[output],
        name="/model/block1.{}/mlp/Block1PackedDwconv".format(block_index),
        domain=PLUGIN_DOMAIN,
        height=height,
        width=width,
        channels=channels,
        packed_weights=packed_weights,
        packed_bias=packed_bias,
        plugin_namespace=PLUGIN_NAMESPACE,
        plugin_version=PLUGIN_VERSION,
    )

    # 步骤 5：在原 Transpose 位置插入单输入插件，只删除布局链和 Conv。
    chain = [transpose_in, reshape_in, conv, reshape_out, transpose_out]
    remove_names = {node.name for node in chain}
    original_nodes = list(model.graph.node)
    insert_at = next(
        index
        for index, node in enumerate(original_nodes)
        if node.name == transpose_in.name
    )
    rewritten: List[NodeProto] = []
    for index, node in enumerate(original_nodes):
        if index == insert_at:
            rewritten.append(plugin)
        if node.name not in remove_names:
            rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    print(
        "[REPLACE] block1.{}: C={} packed_weight_half2={} "
        "packed_bias_half2={}; GELU preserved".format(
            block_index,
            channels,
            len(packed_weights),
            len(packed_bias),
        )
    )


def remove_dead_nodes_and_initializers(model: ModelProto) -> None:
    """从图输出反向保留有效计算，清理旧布局及权重生产支路。"""
    original_nodes = list(model.graph.node)
    producer_index: Dict[str, int] = {}
    for index, node in enumerate(original_nodes):
        for output_name in node.output:
            if output_name:
                producer_index[output_name] = index

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

    used_inputs = {
        input_name
        for node in model.graph.node
        for input_name in node.input
        if input_name
    }
    graph_outputs = {value.name for value in model.graph.output}
    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in used_inputs or initializer.name in graph_outputs
    ]
    removed_initializers = len(model.graph.initializer) - len(kept_initializers)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    if removed_initializers:
        print(
            "[CLEAN  ] removed {} unused initializers".format(
                removed_initializers
            )
        )


def ensure_plugin_opset(model: ModelProto) -> None:
    """确保模型声明 egcinet 自定义 domain。"""
    for opset in model.opset_import:
        if opset.domain == PLUGIN_DOMAIN:
            if opset.version != 1:
                raise RuntimeError(
                    "{} domain uses unsupported opset {}".format(
                        PLUGIN_DOMAIN,
                        opset.version,
                    )
                )
            return
    model.opset_import.append(helper.make_operatorsetid(PLUGIN_DOMAIN, 1))


def validate_rewritten_model(model: ModelProto) -> None:
    """终检插件数量、字段长度、旧节点清理和 GELU 保留情况。"""
    nodes_by_name, _, consumers = build_maps(model)
    expected_plugin_names = {
        "/model/block1.{}/mlp/Block1PackedDwconv".format(index)
        for index in BLOCK_INDICES
    }
    plugins = {
        node.name: node
        for node in model.graph.node
        if node.op_type == PLUGIN_OP and node.domain == PLUGIN_DOMAIN
    }
    if set(plugins) != expected_plugin_names:
        raise RuntimeError(
            "plugin nodes mismatch: expected {}, got {}".format(
                sorted(expected_plugin_names),
                sorted(plugins),
            )
        )

    for block_index in BLOCK_INDICES:
        prefix = "/model/block1.{}/mlp".format(block_index)
        dwconv_prefix = prefix + "/dwconv"
        plugin_name = prefix + "/Block1PackedDwconv"
        plugin = plugins[plugin_name]
        if len(plugin.input) != 1 or len(plugin.output) != 1:
            raise RuntimeError("{} must have one input and one output".format(plugin_name))

        attributes = node_attributes(plugin)
        channels = int(attributes.get("channels", 0))
        packed_weights = list(attributes.get("packed_weights", []))
        packed_bias = list(attributes.get("packed_bias", []))
        if channels <= 0 or (channels & 1) != 0:
            raise RuntimeError("{} has invalid channels".format(plugin_name))
        if len(packed_weights) != 9 * channels // 2 or \
           len(packed_bias) != channels // 2:
            raise RuntimeError("{} packed field length mismatch".format(plugin_name))

        old_names = {
            dwconv_prefix + "/Transpose",
            dwconv_prefix + "/Reshape",
            dwconv_prefix + "/dwconv/Conv",
            dwconv_prefix + "/Reshape_1",
            dwconv_prefix + "/Transpose_1",
        }
        remaining = [name for name in old_names if name in nodes_by_name]
        if remaining:
            raise RuntimeError("old layout nodes remain: {}".format(remaining))

        # GELU 五个主节点必须继续存在，并消费插件输出。
        validate_exact_gelu(
            nodes_by_name,
            consumers,
            prefix,
            plugin.output[0],
        )

    onnx.checker.check_model(model)
    print("[PASS   ] 3/3 packed DWConv plugins inserted; 3/3 GELU graphs preserved")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace block1 layout chains with packed-weight DWConv plugins",
    )
    parser.add_argument(
        "--input",
        default="deploy/egcienet_352_multiclass_qdq_v7_block1_fc1.onnx",
        help="validated V7 Q/DQ ONNX model",
    )
    parser.add_argument(
        "--output",
        default="deploy/egcienet_352_multiclass_qdq_v11_block1_packed_dwconv.onnx",
        help="V11 ONNX model containing packed-weight DWConv plugin nodes",
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
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")
    if not input_path.is_file():
        raise FileNotFoundError("input ONNX not found: {}".format(input_path))
    if input_path.resolve() == output_path.resolve():
        raise RuntimeError("output must not overwrite input ONNX")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "output already exists: {}; pass --force to overwrite".format(
                output_path
            )
        )

    print("[INPUT  ] {}".format(input_path))
    print("[OUTPUT ] {}".format(output_path))
    print("[SHAPE  ] block1 H={} W={}".format(args.height, args.width))
    print("[BASE   ] V7 Q/DQ; static DWConv weights; native GELU preserved")

    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    for block_index in BLOCK_INDICES:
        replace_one_block(model, block_index, args.height, args.width)

    ensure_plugin_opset(model)
    remove_dead_nodes_and_initializers(model)
    validate_rewritten_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    saved_model = onnx.load(str(output_path))
    validate_rewritten_model(saved_model)
    print("[DONE   ] {}".format(output_path))


if __name__ == "__main__":
    main()
