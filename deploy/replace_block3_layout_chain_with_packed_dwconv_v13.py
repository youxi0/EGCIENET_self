#!/usr/bin/env python3

"""
V13：在 V12 的基础上替换 block3.0～block3.17 的 MLP 布局链和 DWConv。

每个 block3 中原始计算为：

    Transpose -> Reshape -> DWConv -> Reshape -> Transpose -> exact-erf GELU

V13 只把前五个节点替换为 EGCINET_Block1PackedDwconv Plugin，并显式设置
fuse_gelu=0。Plugin 直接接收、输出 [B, H*W, C] token-major FP16 tensor，
在 CUDA kernel 内按二维坐标访问数据，不再物化两侧的布局转换；原 exact-erf
GELU 子图完整保留，继续交给 TensorRT/Myelin 优化。

本脚本以已经包含三个 block1 Plugin 的 V12 ONNX 为输入，不会覆盖 V12，也不
修改 fc1/fc2 的 Q/DQ。V14 会复用本文件的公共改图逻辑生成融合 GELU 的版本。
"""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import onnx
from onnx import ModelProto, NodeProto, helper

import replace_block1_layout_chain_with_packed_dwconv_v11 as v11
import replace_block1_layout_chain_with_packed_dwconv_gelu_v12 as v12


BLOCK3_INDICES = tuple(range(18))
DEFAULT_HEIGHT = 22
DEFAULT_WIDTH = 22
BLOCK1_PLUGIN_NAMES = {
    "/model/block1.{}/mlp/Block1PackedDwconvGelu".format(index)
    for index in v12.BLOCK_INDICES
}


def locate_dwconv_chain(
    nodes_by_name: Dict[str, NodeProto],
    consumers: Dict[str, List[NodeProto]],
    prefix: str,
) -> Tuple[NodeProto, NodeProto, NodeProto, NodeProto, NodeProto]:
    """精确定位并验证一个 block3 的五节点布局/DWConv 主链。"""
    dwconv_prefix = prefix + "/dwconv"
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

    if len(transpose_in.input) != 1:
        raise RuntimeError("{} must have one input".format(transpose_in.name))
    if len(conv.input) < 3:
        raise RuntimeError(
            "{} must have activation, weight and bias".format(conv.name)
        )
    if len(transpose_out.output) != 1:
        raise RuntimeError("{} must have one output".format(transpose_out.name))

    transpose_in_perm = list(
        v11.node_attributes(transpose_in).get("perm", [])
    )
    transpose_out_perm = list(
        v11.node_attributes(transpose_out).get("perm", [])
    )
    if transpose_in_perm != [0, 2, 1] or transpose_out_perm != [0, 2, 1]:
        raise RuntimeError(
            "{} DWConv transposes must both use [0,2,1]".format(prefix)
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

    # Conv 输出还可能被导出图中的 Shape 节点读取；该支路在替换后会被死图清理。
    v11.require_direct_edge(transpose_in, reshape_in, consumers)
    v11.require_direct_edge(reshape_in, conv, consumers)
    v11.require_direct_edge(
        conv,
        reshape_out,
        consumers,
        allowed_extra_consumers=((dwconv_prefix + "/Shape", "Shape"),),
    )
    v11.require_direct_edge(reshape_out, transpose_out, consumers)
    return transpose_in, reshape_in, conv, reshape_out, transpose_out


def replace_one_block3(
    model: ModelProto,
    block_index: int,
    height: int,
    width: int,
    fuse_gelu: bool,
) -> None:
    """替换一个 block3；fuse_gelu 决定是否同时删除并融合原 GELU。"""
    nodes_by_name, producer, consumers = v11.build_maps(model)
    initializers = {
        initializer.name: initializer
        for initializer in model.graph.initializer
    }
    prefix = "/model/block3.{}/mlp".format(block_index)
    (
        transpose_in,
        reshape_in,
        conv,
        reshape_out,
        transpose_out,
    ) = locate_dwconv_chain(nodes_by_name, consumers, prefix)

    # 步骤 1：沿静态参数链读取实际权重，并复用 V11 的 [9,C/2] half2 打包。
    weight = v11.resolve_static_tensor(
        conv.input[1], producer, initializers
    )
    bias = v11.resolve_static_tensor(
        conv.input[2], producer, initializers
    )
    channels, packed_weights, packed_bias = v11.pack_dwconv_parameters(
        weight,
        bias,
        conv.name,
    )
    group = int(v11.node_attributes(conv).get("group", 1))
    if group != channels:
        raise RuntimeError(
            "{} group {} does not match channels {}".format(
                conv.name, group, channels
            )
        )
    # CUDA 实现使用一个线程处理一个 half2 通道对，当前上限是 1024 对。
    if channels // 2 > 1024:
        raise RuntimeError(
            "{} channels exceed current Plugin kernel limit: {}".format(
                conv.name, channels
            )
        )

    # 步骤 2：确定 Plugin 输出边界。非融合版复用 Transpose_1 输出，因而原
    # GELU 无需改线；融合版复用 Mul_1 输出，并删除完整 exact-erf 子图。
    activation = transpose_in.input[0]
    gelu_nodes: Sequence[NodeProto] = ()
    if fuse_gelu:
        gelu_nodes, gelu_output = v12.locate_exact_gelu_chain(
            nodes_by_name,
            consumers,
            prefix,
            transpose_out.output[0],
        )
        output = gelu_output.output[0]
        plugin_suffix = "Block3PackedDwconvGelu"
    else:
        output = transpose_out.output[0]
        v11.validate_exact_gelu(
            nodes_by_name,
            consumers,
            prefix,
            output,
        )
        plugin_suffix = "Block3PackedDwconv"

    plugin = helper.make_node(
        v11.PLUGIN_OP,
        inputs=[activation],
        outputs=[output],
        name=prefix + "/" + plugin_suffix,
        domain=v11.PLUGIN_DOMAIN,
        height=height,
        width=width,
        channels=channels,
        fuse_gelu=int(fuse_gelu),
        packed_weights=packed_weights,
        packed_bias=packed_bias,
        plugin_namespace=v11.PLUGIN_NAMESPACE,
        plugin_version=v11.PLUGIN_VERSION,
    )

    # 步骤 3：在原输入 Transpose 位置插入 Plugin，并删除被替换的主链。
    chain: Sequence[NodeProto] = (
        transpose_in,
        reshape_in,
        conv,
        reshape_out,
        transpose_out,
        *gelu_nodes,
    )
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
        "[REPLACE] block3.{}: H={} W={} C={} packed_weight_half2={} "
        "packed_bias_half2={}; GELU {}".format(
            block_index,
            height,
            width,
            channels,
            len(packed_weights),
            len(packed_bias),
            "fused" if fuse_gelu else "preserved",
        )
    )


def validate_plugin_fields(
    plugin: NodeProto,
    expected_height: int,
    expected_width: int,
    expected_fuse_gelu: int,
) -> None:
    """检查一个 packed DWConv Plugin 的形状、开关和打包字段长度。"""
    if len(plugin.input) != 1 or len(plugin.output) != 1:
        raise RuntimeError(
            "{} must have one input and one output".format(plugin.name)
        )
    attributes = v11.node_attributes(plugin)
    height = int(attributes.get("height", 0))
    width = int(attributes.get("width", 0))
    channels = int(attributes.get("channels", 0))
    fuse_gelu = int(attributes.get("fuse_gelu", 0))
    if height != expected_height or width != expected_width:
        raise RuntimeError(
            "{} shape mismatch: expected {}x{}, got {}x{}".format(
                plugin.name,
                expected_height,
                expected_width,
                height,
                width,
            )
        )
    if channels <= 0 or (channels & 1) != 0:
        raise RuntimeError("{} has invalid channels".format(plugin.name))
    if fuse_gelu != expected_fuse_gelu:
        raise RuntimeError(
            "{} fuse_gelu mismatch: expected {}, got {}".format(
                plugin.name,
                expected_fuse_gelu,
                fuse_gelu,
            )
        )
    if len(list(attributes.get("packed_weights", []))) != 9 * channels // 2 or \
       len(list(attributes.get("packed_bias", []))) != channels // 2:
        raise RuntimeError(
            "{} packed field length mismatch".format(plugin.name)
        )


def validate_rewritten_model(
    model: ModelProto,
    height: int,
    width: int,
    fuse_gelu: bool,
) -> None:
    """终检 3 个 block1 与 18 个 block3 Plugin，以及旧节点清理结果。"""
    nodes_by_name, _, consumers = v11.build_maps(model)
    block3_suffix = (
        "Block3PackedDwconvGelu" if fuse_gelu else "Block3PackedDwconv"
    )
    block3_plugin_names = {
        "/model/block3.{}/mlp/{}".format(index, block3_suffix)
        for index in BLOCK3_INDICES
    }
    expected_names = BLOCK1_PLUGIN_NAMES | block3_plugin_names
    plugins = {
        node.name: node
        for node in model.graph.node
        if node.op_type == v11.PLUGIN_OP and node.domain == v11.PLUGIN_DOMAIN
    }
    if set(plugins) != expected_names:
        raise RuntimeError(
            "plugin nodes mismatch: expected {}, got {}".format(
                sorted(expected_names),
                sorted(plugins),
            )
        )

    # 输入 V12 的 block1 Plugin 必须保持原样，并继续融合 GELU。
    for block_index in v12.BLOCK_INDICES:
        plugin_name = (
            "/model/block1.{}/mlp/Block1PackedDwconvGelu".format(block_index)
        )
        validate_plugin_fields(plugins[plugin_name], 88, 88, 1)

    # 新增的 18 个 block3 Plugin 必须具有统一的 H/W 和融合开关。
    for block_index in BLOCK3_INDICES:
        prefix = "/model/block3.{}/mlp".format(block_index)
        dwconv_prefix = prefix + "/dwconv"
        plugin_name = prefix + "/" + block3_suffix
        plugin = plugins[plugin_name]
        validate_plugin_fields(plugin, height, width, int(fuse_gelu))

        old_layout_names = {
            dwconv_prefix + "/Transpose",
            dwconv_prefix + "/Reshape",
            dwconv_prefix + "/dwconv/Conv",
            dwconv_prefix + "/Reshape_1",
            dwconv_prefix + "/Transpose_1",
        }
        remaining_layout = sorted(
            name for name in old_layout_names if name in nodes_by_name
        )
        if remaining_layout:
            raise RuntimeError(
                "old block3 layout nodes remain: {}".format(remaining_layout)
            )

        if fuse_gelu:
            old_gelu_names = {
                prefix + "/act/Div",
                prefix + "/act/Erf",
                prefix + "/act/Add",
                prefix + "/act/Mul",
                prefix + "/act/Mul_1",
            }
            remaining_gelu = sorted(
                name for name in old_gelu_names if name in nodes_by_name
            )
            if remaining_gelu:
                raise RuntimeError(
                    "old block3 GELU nodes remain: {}".format(remaining_gelu)
                )
        else:
            v11.validate_exact_gelu(
                nodes_by_name,
                consumers,
                prefix,
                plugin.output[0],
            )

    onnx.checker.check_model(model)
    print(
        "[PASS   ] 3 block1 + 18 block3 plugins validated; block3 GELU {}".format(
            "fused" if fuse_gelu else "preserved"
        )
    )


def generate_variant(
    input_path: Path,
    output_path: Path,
    height: int,
    width: int,
    fuse_gelu: bool,
    force: bool,
) -> None:
    """加载 V12、生成指定融合模式并在保存前后各校验一次。"""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not input_path.is_file():
        raise FileNotFoundError("input V12 ONNX not found: {}".format(input_path))
    if input_path.resolve() == output_path.resolve():
        raise RuntimeError("output must not overwrite input ONNX")
    if output_path.exists() and not force:
        raise FileExistsError(
            "output already exists: {}; pass --force to overwrite".format(
                output_path
            )
        )

    print("[INPUT  ] {}".format(input_path))
    print("[OUTPUT ] {}".format(output_path))
    print("[SHAPE  ] block3 H={} W={}".format(height, width))
    print(
        "[MODE   ] block3 packed DWConv; GELU {}".format(
            "fused with FP32 tanh approximation" if fuse_gelu
            else "preserved as TensorRT exact-erf graph"
        )
    )

    model = onnx.load(str(input_path))
    # 先确认输入确实是完整的 V12，而不是已经二次改写的其他实验模型。
    v12.validate_rewritten_model(model)
    for block_index in BLOCK3_INDICES:
        replace_one_block3(
            model,
            block_index,
            height,
            width,
            fuse_gelu,
        )

    v11.ensure_plugin_opset(model)
    v11.remove_dead_nodes_and_initializers(model)
    validate_rewritten_model(model, height, width, fuse_gelu)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    saved_model = onnx.load(str(output_path))
    validate_rewritten_model(saved_model, height, width, fuse_gelu)
    print("[DONE   ] {}".format(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V13: replace block3 layout/DWConv chains and preserve TensorRT GELU"
        ),
    )
    parser.add_argument(
        "--input",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v12_"
            "block1_packed_dwconv_gelu.onnx"
        ),
        help="validated V12 ONNX containing three block1 plugins",
    )
    parser.add_argument(
        "--output",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v13_"
            "block3_packed_dwconv.onnx"
        ),
        help="V13 ONNX with 18 non-GELU block3 plugins",
    )
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_variant(
        Path(args.input),
        Path(args.output),
        args.height,
        args.width,
        fuse_gelu=False,
        force=args.force,
    )


if __name__ == "__main__":
    main()
