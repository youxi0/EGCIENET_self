#!/usr/bin/env python3

"""
V12：在 V11 packed-weight DWConv 的基础上重新融合 exact GELU。

替换范围：

    Transpose -> Reshape -> DWConv -> Reshape -> Transpose
      -> Div -> [Cast] -> Erf -> [Cast] -> Add -> Mul -> Mul_1

权重和 bias 仍由 V11 公共函数离线打包为 half2 位模式；新节点增加
fuse_gelu=1，要求 Block1PackedDwconvPlugin 在同一个 CUDA kernel 内完成
DWConv 和 exact GELU。必须从 V7 生成，不能在 V11 自定义节点模型上继续改图。
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import onnx
from onnx import ModelProto, NodeProto, helper

import replace_block1_layout_chain_with_packed_dwconv_v11 as v11


BLOCK_INDICES = v11.BLOCK_INDICES


def locate_exact_gelu_chain(
    nodes_by_name: Dict[str, NodeProto],
    consumers: Dict[str, List[NodeProto]],
    prefix: str,
    gelu_input_tensor: str,
) -> Tuple[List[NodeProto], NodeProto]:
    """验证 exact GELU 主链并返回需要融合的全部节点和最终输出节点。"""
    act_prefix = prefix + "/act"
    div = v11.require_node(nodes_by_name, act_prefix + "/Div", "Div", prefix)
    erf = v11.require_node(nodes_by_name, act_prefix + "/Erf", "Erf", prefix)
    add = v11.require_node(nodes_by_name, act_prefix + "/Add", "Add", prefix)
    mul = v11.require_node(nodes_by_name, act_prefix + "/Mul", "Mul", prefix)
    mul_final = v11.require_node(
        nodes_by_name,
        act_prefix + "/Mul_1",
        "Mul",
        prefix,
    )

    v11.require_consumer_names(
        gelu_input_tensor,
        (div.name, mul.name),
        consumers,
    )
    casts_before_erf = v11.trace_optional_casts(div, erf, consumers)
    casts_after_erf = v11.trace_optional_casts(erf, add, consumers)
    v11.require_direct_edge(add, mul, consumers)
    if gelu_input_tensor not in mul.input:
        raise RuntimeError("{} must also consume GELU x".format(mul.name))
    v11.require_direct_edge(mul, mul_final, consumers)
    if len(mul_final.output) != 1:
        raise RuntimeError("{} must have one output".format(mul_final.name))

    return (
        [
            div,
            *casts_before_erf,
            erf,
            *casts_after_erf,
            add,
            mul,
            mul_final,
        ],
        mul_final,
    )


def replace_one_block(
    model: ModelProto,
    block_index: int,
    height: int,
    width: int,
) -> None:
    """将一个 block1 的布局链、DWConv 和 exact GELU 替换为一个 V12 节点。"""
    nodes_by_name, producer, consumers = v11.build_maps(model)
    initializers = {
        initializer.name: initializer
        for initializer in model.graph.initializer
    }
    prefix = "/model/block1.{}/mlp".format(block_index)
    dwconv_prefix = prefix + "/dwconv"

    # 步骤 1：定位并验证 DWConv 两侧布局链。
    transpose_in = v11.require_node(
        nodes_by_name, dwconv_prefix + "/Transpose", "Transpose", prefix
    )
    reshape_in = v11.require_node(
        nodes_by_name, dwconv_prefix + "/Reshape", "Reshape", prefix
    )
    conv = v11.require_node(
        nodes_by_name, dwconv_prefix + "/dwconv/Conv", "Conv", prefix
    )
    reshape_out = v11.require_node(
        nodes_by_name, dwconv_prefix + "/Reshape_1", "Reshape", prefix
    )
    transpose_out = v11.require_node(
        nodes_by_name, dwconv_prefix + "/Transpose_1", "Transpose", prefix
    )

    if len(transpose_in.input) != 1 or len(conv.input) < 3 or \
       len(transpose_out.output) != 1:
        raise RuntimeError("block1.{} DWConv interface mismatch".format(block_index))
    if list(v11.node_attributes(transpose_in).get("perm", [])) != [0, 2, 1] or \
       list(v11.node_attributes(transpose_out).get("perm", [])) != [0, 2, 1]:
        raise RuntimeError(
            "block1.{} DWConv transposes must use [0,2,1]".format(block_index)
        )

    conv_attributes = v11.node_attributes(conv)
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

    v11.require_direct_edge(transpose_in, reshape_in, consumers)
    v11.require_direct_edge(reshape_in, conv, consumers)
    v11.require_direct_edge(
        conv,
        reshape_out,
        consumers,
        allowed_extra_consumers=((dwconv_prefix + "/Shape", "Shape"),),
    )
    v11.require_direct_edge(reshape_out, transpose_out, consumers)

    # 步骤 2：从 V7 静态参数链读取权重并沿用 V11 的 [9,C/2] half2 打包。
    weight = v11.resolve_static_tensor(conv.input[1], producer, initializers)
    bias = v11.resolve_static_tensor(conv.input[2], producer, initializers)
    channels, packed_weights, packed_bias = v11.pack_dwconv_parameters(
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

    # 步骤 3：验证 GELU，并把插件输出边界移动到原 Mul_1 输出。
    gelu_nodes, mul_final = locate_exact_gelu_chain(
        nodes_by_name,
        consumers,
        prefix,
        transpose_out.output[0],
    )
    activation = transpose_in.input[0]
    output = mul_final.output[0]

    plugin = helper.make_node(
        v11.PLUGIN_OP,
        inputs=[activation],
        outputs=[output],
        name="/model/block1.{}/mlp/Block1PackedDwconvGelu".format(block_index),
        domain=v11.PLUGIN_DOMAIN,
        height=height,
        width=width,
        channels=channels,
        fuse_gelu=1,
        packed_weights=packed_weights,
        packed_bias=packed_bias,
        plugin_namespace=v11.PLUGIN_NAMESPACE,
        plugin_version=v11.PLUGIN_VERSION,
    )

    # 步骤 4：删除布局链、DWConv、GELU 和可选 Cast，保留最终 tensor 名称。
    chain = [
        transpose_in,
        reshape_in,
        conv,
        reshape_out,
        transpose_out,
        *gelu_nodes,
    ]
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
        "packed_bias_half2={}; exact GELU fused".format(
            block_index,
            channels,
            len(packed_weights),
            len(packed_bias),
        )
    )


def validate_rewritten_model(model: ModelProto) -> None:
    """检查三个 V12 节点、packed 字段和旧 DWConv/GELU 节点清理结果。"""
    nodes_by_name, _, _ = v11.build_maps(model)
    expected_names = {
        "/model/block1.{}/mlp/Block1PackedDwconvGelu".format(index)
        for index in BLOCK_INDICES
    }
    plugins = {
        node.name: node
        for node in model.graph.node
        if node.op_type == v11.PLUGIN_OP and node.domain == v11.PLUGIN_DOMAIN
    }
    if set(plugins) != expected_names:
        raise RuntimeError(
            "V12 plugin nodes mismatch: expected {}, got {}".format(
                sorted(expected_names),
                sorted(plugins),
            )
        )

    for block_index in BLOCK_INDICES:
        prefix = "/model/block1.{}/mlp".format(block_index)
        plugin_name = prefix + "/Block1PackedDwconvGelu"
        plugin = plugins[plugin_name]
        if len(plugin.input) != 1 or len(plugin.output) != 1:
            raise RuntimeError("{} must have one input and one output".format(plugin_name))

        attributes = v11.node_attributes(plugin)
        channels = int(attributes.get("channels", 0))
        if int(attributes.get("fuse_gelu", 0)) != 1:
            raise RuntimeError("{} must set fuse_gelu=1".format(plugin_name))
        if len(list(attributes.get("packed_weights", []))) != 9 * channels // 2 or \
           len(list(attributes.get("packed_bias", []))) != channels // 2:
            raise RuntimeError("{} packed field length mismatch".format(plugin_name))

        old_names = {
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
        remaining = sorted(name for name in old_names if name in nodes_by_name)
        if remaining:
            raise RuntimeError("old DWConv/GELU nodes remain: {}".format(remaining))

    onnx.checker.check_model(model)
    print("[PASS   ] 3/3 packed DWConv + exact GELU chains fused")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse block1 packed DWConv and exact GELU into V12 plugin nodes",
    )
    parser.add_argument(
        "--input",
        default="deploy/egcienet_352_multiclass_qdq_v7_block1_fc1.onnx",
    )
    parser.add_argument(
        "--output",
        default="deploy/egcienet_352_multiclass_qdq_v12_block1_packed_dwconv_gelu.onnx",
    )
    parser.add_argument("--height", type=int, default=88)
    parser.add_argument("--width", type=int, default=88)
    parser.add_argument("--force", action="store_true")
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
            "output already exists: {}; pass --force to overwrite".format(output_path)
        )

    print("[INPUT  ] {}".format(input_path))
    print("[OUTPUT ] {}".format(output_path))
    print("[SHAPE  ] block1 H={} W={}".format(args.height, args.width))
    print("[BASE   ] V7 Q/DQ; packed DWConv + exact GELU")

    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    for block_index in BLOCK_INDICES:
        replace_one_block(model, block_index, args.height, args.width)

    v11.ensure_plugin_opset(model)
    v11.remove_dead_nodes_and_initializers(model)
    validate_rewritten_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    saved_model = onnx.load(str(output_path))
    validate_rewritten_model(saved_model)
    print("[DONE   ] {}".format(output_path))


if __name__ == "__main__":
    main()
