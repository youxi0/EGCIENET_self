#!/usr/bin/env python3

"""
V19：从稳定的 V16 ONNX 重新分叉，只量化四个 encoder stage 的全部
Attention Q 投影。

目标范围：
  - block1.0～2：3个 attn/q/MatMul；
  - block2.0～3：4个 attn/q/MatMul；
  - block3.0～17：18个 attn/q/MatMul；
  - block4.0～2：3个 attn/q/MatMul；
  - 合计28个。

每个目标使用与现有 block3 fc1 已验证 INT8 路径相同的显式 Q/DQ：

  norm1 输出 -> 激活 Q -> 激活 DQ --+
                                      +-> q/MatMul -> FP16 Attention 核心
  q 静态权重 -> 权重 Q -> 权重 DQ --+

约束：
  - 激活 Q/DQ 为每个 q/MatMul 的专用分支，不污染 SR/KV 分支；
  - 权重按 axis=1 做逐输出通道对称 INT8 量化；
  - KV、SR、QK^T、Softmax、Attention×V、projection、残差均不修改；
  - V16 中已有的 block1/2/3/4 Plugin 和全部非目标节点保持不变；
  - 不接受已经做过 Attention 量化的 V17/V18 ONNX 作为输入。
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper

import insert_block3_attn_q_matmul_qdq_v17 as v17


BLOCK_DEPTHS: Dict[int, int] = {
    1: 3,
    2: 4,
    3: 18,
    4: 3,
}

TARGET_SPECS: List[Tuple[int, int, str]] = [
    (
        stage,
        layer_index,
        "/model/block{}.{}/attn/q/MatMul".format(stage, layer_index),
    )
    for stage, depth in BLOCK_DEPTHS.items()
    for layer_index in range(depth)
]

Q_TARGETS: List[str] = [name for _, _, name in TARGET_SPECS]


def make_qdq_nodes(
    model: ModelProto,
    stage: int,
    layer_index: int,
    target: NodeProto,
    producer: Mapping[str, NodeProto],
    consumers: Mapping[str, Sequence[NodeProto]],
    initializers: Mapping[str, TensorProto],
    cache_scales: Mapping[str, float],
    template: Mapping[str, object],
) -> Tuple[List[NodeProto], List[TensorProto]]:
    """为一个 Attention q/MatMul 创建专用激活和权重 Q/DQ。"""
    if target.op_type != "MatMul" or len(target.input) != 2:
        raise RuntimeError("{} 不是合法 MatMul".format(target.name))

    raw_activation = target.input[0]
    raw_weight = target.input[1]

    # V19 只接受未量化的 V16 Attention，避免重复插入或叠加旧实验。
    activation_parent = producer.get(raw_activation)
    weight_parent = producer.get(raw_weight)
    if activation_parent is not None and activation_parent.op_type == "DequantizeLinear":
        raise RuntimeError("{} 激活已经量化".format(target.name))
    if weight_parent is not None and weight_parent.op_type == "DequantizeLinear":
        raise RuntimeError("{} 权重已经量化".format(target.name))

    weight_initializer = initializers.get(raw_weight)
    if weight_initializer is None:
        raise RuntimeError(
            "{} 权重不是直接 initializer: {}".format(
                target.name,
                raw_weight,
            )
        )
    weight_consumers = list(consumers.get(raw_weight, []))
    if len(weight_consumers) != 1 or weight_consumers[0].name != target.name:
        raise RuntimeError(
            "{} 权重不是专用权重: {}".format(
                target.name,
                [node.name for node in weight_consumers],
            )
        )

    weight = np.asarray(numpy_helper.to_array(weight_initializer))
    if weight.ndim != 2:
        raise RuntimeError(
            "{} 权重不是二维: {}".format(target.name, weight.shape)
        )
    output_channels = int(weight.shape[1])

    # 步骤1：每层使用自己的 norm1 输出校准 scale，不能跨 stage 复用。
    cache_key, activation_scale = v17.find_activation_scale(
        raw_activation,
        producer,
        cache_scales,
    )

    # 步骤2：MatMul 权重为 [K,N]，沿 K 轴求每个输出通道的对称 scale。
    weight_fp32 = weight.astype(np.float32, copy=False)
    weight_amax = np.max(np.abs(weight_fp32), axis=0)
    weight_scale = np.maximum(
        weight_amax / np.float32(127.0),
        np.float32(1.0e-8),
    )
    if weight_scale.shape != (output_channels,):
        raise RuntimeError("{} 权重 scale shape 错误".format(target.name))
    if not np.all(np.isfinite(weight_scale)):
        raise RuntimeError("{} 权重 scale 存在非有限值".format(target.name))

    activation_prefix = target.name + "_V19_Activation"
    weight_prefix = target.name + "_V19_Weight"

    activation_scale_q_name = activation_prefix + "_scale_1"
    activation_scale_dq_name = activation_prefix + "_scale_2"
    activation_zero_q_name = activation_prefix + "_zero_point_1"
    activation_zero_dq_name = activation_prefix + "_zero_point_2"
    activation_q_output = activation_prefix + "_QuantizeLinear_Output"
    activation_dq_output = activation_prefix + "_DequantizeLinear_Output"
    activation_q_name = activation_prefix + "_QuantizeLinear"
    activation_dq_name = activation_prefix + "_DequantizeLinear"

    weight_scale_q_name = weight_prefix + "_scale_1"
    weight_scale_dq_name = weight_prefix + "_scale_2"
    weight_zero_q_name = weight_prefix + "_zero_point_1"
    weight_zero_dq_name = weight_prefix + "_zero_point_2"
    weight_q_output = weight_prefix + "_QuantizeLinear_Output"
    weight_dq_output = weight_prefix + "_DequantizeLinear_Output"
    weight_q_name = weight_prefix + "_QuantizeLinear"
    weight_dq_name = weight_prefix + "_DequantizeLinear"

    v17.require_available_names(
        model,
        [
            activation_q_name,
            activation_dq_name,
            weight_q_name,
            weight_dq_name,
        ],
        [
            activation_scale_q_name,
            activation_scale_dq_name,
            activation_zero_q_name,
            activation_zero_dq_name,
            activation_q_output,
            activation_dq_output,
            weight_scale_q_name,
            weight_scale_dq_name,
            weight_zero_q_name,
            weight_zero_dq_name,
            weight_q_output,
            weight_dq_output,
        ],
    )

    activation_scale_value = np.asarray(activation_scale).reshape(())
    scalar_zero = np.asarray(0, dtype=np.int8).reshape(())
    channel_zero = np.zeros((output_channels,), dtype=np.int8)

    new_initializers = [
        v17.make_initializer_like(
            template["activation_scale"],
            activation_scale_q_name,
            activation_scale_value,
        ),
        v17.make_initializer_like(
            template["activation_scale"],
            activation_scale_dq_name,
            activation_scale_value,
        ),
        v17.make_initializer_like(
            template["activation_zero"],
            activation_zero_q_name,
            scalar_zero,
        ),
        v17.make_initializer_like(
            template["activation_zero"],
            activation_zero_dq_name,
            scalar_zero,
        ),
        v17.make_initializer_like(
            template["weight_scale"],
            weight_scale_q_name,
            weight_scale,
        ),
        v17.make_initializer_like(
            template["weight_scale"],
            weight_scale_dq_name,
            weight_scale,
        ),
        v17.make_initializer_like(
            template["weight_zero"],
            weight_zero_q_name,
            channel_zero,
        ),
        v17.make_initializer_like(
            template["weight_zero"],
            weight_zero_dq_name,
            channel_zero,
        ),
    ]

    # 步骤3：结构与已成功的 fc1 路径一致；激活为 per-tensor，权重为 axis=1。
    activation_q = helper.make_node(
        "QuantizeLinear",
        inputs=[
            raw_activation,
            activation_scale_q_name,
            activation_zero_q_name,
        ],
        outputs=[activation_q_output],
        name=activation_q_name,
    )
    activation_dq = helper.make_node(
        "DequantizeLinear",
        inputs=[
            activation_q_output,
            activation_scale_dq_name,
            activation_zero_dq_name,
        ],
        outputs=[activation_dq_output],
        name=activation_dq_name,
    )
    weight_q = helper.make_node(
        "QuantizeLinear",
        inputs=[raw_weight, weight_scale_q_name, weight_zero_q_name],
        outputs=[weight_q_output],
        name=weight_q_name,
        axis=1,
    )
    weight_dq = helper.make_node(
        "DequantizeLinear",
        inputs=[weight_q_output, weight_scale_dq_name, weight_zero_dq_name],
        outputs=[weight_dq_output],
        name=weight_dq_name,
        axis=1,
    )

    # 步骤4：只改 q/MatMul 两个输入；输出仍为高精度并进入原 Attention 核心。
    target.input[0] = activation_dq_output
    target.input[1] = weight_dq_output

    print(
        "[INSERT] block{}.{} q/MatMul: activation_scale={:.9g}, "
        "cache_key={}, weight_shape={}".format(
            stage,
            layer_index,
            activation_scale,
            cache_key,
            tuple(weight.shape),
        )
    )
    return (
        [activation_q, activation_dq, weight_q, weight_dq],
        new_initializers,
    )


def validate_result(
    model: ModelProto,
    before_qdq: Counter,
    before_plugins: Mapping[str, str],
) -> None:
    """验证只有28个 Q 投影具有完整、专用的激活和权重 Q/DQ。"""
    onnx.checker.check_model(model)
    nodes_by_name, producer, consumers, initializers = v17.build_maps(model)

    quantized_attention_matmuls = {
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and "/attn/" in node.name
        and v17.is_quantized_weight_matmul(node, producer)
    }
    expected = set(Q_TARGETS)
    missing = sorted(expected - quantized_attention_matmuls)
    unexpected = sorted(quantized_attention_matmuls - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Attention MatMul 量化集合错误:\n  missing={}\n  unexpected={}".format(
                missing,
                unexpected,
            )
        )

    stage_counts: Counter = Counter()
    for stage, layer_index, target_name in TARGET_SPECS:
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        activation_dq = producer.get(target.input[0])
        weight_dq = producer.get(target.input[1])
        if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
            raise RuntimeError("{} 激活输入不是直接 DQ".format(target_name))
        if weight_dq is None or weight_dq.op_type != "DequantizeLinear":
            raise RuntimeError("{} 权重输入不是直接 DQ".format(target_name))

        activation_q = producer.get(activation_dq.input[0])
        weight_q = producer.get(weight_dq.input[0])
        if activation_q is None or activation_q.op_type != "QuantizeLinear":
            raise RuntimeError("{} 激活缺少 Q -> DQ".format(target_name))
        if weight_q is None or weight_q.op_type != "QuantizeLinear":
            raise RuntimeError("{} 权重缺少 Q -> DQ".format(target_name))

        dq_consumers = consumers.get(activation_dq.output[0], [])
        if len(dq_consumers) != 1 or dq_consumers[0].name != target_name:
            raise RuntimeError(
                "{} 激活 DQ 不是专用分支: {}".format(
                    target_name,
                    [node.name for node in dq_consumers],
                )
            )

        if v17.get_axis(weight_q) != 1 or v17.get_axis(weight_dq) != 1:
            raise RuntimeError("{} 权重 Q/DQ axis 不是1".format(target_name))

        raw_weight = weight_q.input[0]
        weight = v17.initializer_array(initializers, raw_weight)
        scale_q = v17.initializer_array(initializers, weight_q.input[1])
        scale_dq = v17.initializer_array(initializers, weight_dq.input[1])
        zero_q = v17.initializer_array(initializers, weight_q.input[2])
        zero_dq = v17.initializer_array(initializers, weight_dq.input[2])
        if scale_q.shape != (weight.shape[1],):
            raise RuntimeError("{} 权重 scale shape 错误".format(target_name))
        if scale_dq.shape != scale_q.shape or not np.array_equal(scale_q, scale_dq):
            raise RuntimeError("{} 权重 Q/DQ scale 不一致".format(target_name))
        if zero_q.dtype != np.int8 or zero_dq.dtype != np.int8:
            raise RuntimeError("{} 权重 zero-point 不是 INT8".format(target_name))
        if np.any(zero_q != 0) or np.any(zero_dq != 0):
            raise RuntimeError("{} 权重 zero-point 不是全零".format(target_name))

        activation_scale_q = v17.initializer_array(
            initializers,
            activation_q.input[1],
        )
        activation_scale_dq = v17.initializer_array(
            initializers,
            activation_dq.input[1],
        )
        if activation_scale_q.shape != () or activation_scale_dq.shape != ():
            raise RuntimeError("{} 激活 scale 不是标量".format(target_name))
        if not np.array_equal(activation_scale_q, activation_scale_dq):
            raise RuntimeError("{} 激活 Q/DQ scale 不一致".format(target_name))

        # KV 和 projection 必须继续保持未量化。
        for suffix in ("kv/MatMul", "proj/MatMul"):
            other_name = "/model/block{}.{}/attn/{}".format(
                stage,
                layer_index,
                suffix,
            )
            other = v17.require_node(nodes_by_name, other_name, "MatMul")
            if v17.is_quantized_weight_matmul(other, producer):
                raise RuntimeError("出现非目标 Attention 量化: {}".format(other_name))

        stage_counts[stage] += 1

    after_qdq = v17.qdq_count(model)
    expected_added_each = len(Q_TARGETS) * 2
    if after_qdq["QuantizeLinear"] - before_qdq["QuantizeLinear"] != expected_added_each:
        raise RuntimeError(
            "新增 QuantizeLinear 数量不是{}: before={}, after={}".format(
                expected_added_each,
                before_qdq,
                after_qdq,
            )
        )
    if after_qdq["DequantizeLinear"] - before_qdq["DequantizeLinear"] != expected_added_each:
        raise RuntimeError(
            "新增 DequantizeLinear 数量不是{}: before={}, after={}".format(
                expected_added_each,
                before_qdq,
                after_qdq,
            )
        )

    after_plugins = v17.plugin_snapshot(model)
    if dict(before_plugins) != after_plugins:
        raise RuntimeError("V16 的 egcinet Plugin 节点发生变化")

    expected_stage_counts = Counter(BLOCK_DEPTHS)
    if stage_counts != expected_stage_counts:
        raise RuntimeError(
            "stage 目标数量错误: expected={}, actual={}".format(
                expected_stage_counts,
                stage_counts,
            )
        )

    print("\n[VALIDATE]")
    print("  block1 attention q targets       : 3/3")
    print("  block2 attention q targets       : 4/4")
    print("  block3 attention q targets       : 18/18")
    print("  block4 attention q targets       : 3/3")
    print("  total q INT8 targets             : 28/28")
    print("  dedicated activation Q/DQ        : 28/28")
    print("  per-channel weight Q/DQ axis=1   : 28/28")
    print("  KV/projection quantized           : 0")
    print("  added QuantizeLinear              : {}".format(expected_added_each))
    print("  added DequantizeLinear            : {}".format(expected_added_each))
    print("  egcinet Plugin nodes unchanged    : {}".format(len(after_plugins)))


def rewrite_model(
    model: ModelProto,
    cache_scales: Mapping[str, float],
) -> Tuple[Counter, Dict[str, str]]:
    """从干净 V16 为28个 Q 投影插入 Q/DQ。"""
    onnx.checker.check_model(model)
    before_qdq = v17.qdq_count(model)
    before_plugins = v17.plugin_snapshot(model)
    nodes_by_name, producer, consumers, initializers = v17.build_maps(model)

    existing_attention_quantized = sorted(
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and "/attn/" in node.name
        and v17.is_quantized_weight_matmul(node, producer)
    )
    if existing_attention_quantized:
        raise RuntimeError(
            "输入不是干净 V16，已经存在 Attention 权重量化: {}".format(
                existing_attention_quantized
            )
        )

    # 只复用 block3.0 fc1 已验证 Q/DQ 的 dtype、INT8 零点和 axis 格式。
    # 数值 scale 不复用；每个 Attention 目标仍使用自己的 cache/weight scale。
    template = v17.inspect_fc1_template(
        0,
        nodes_by_name,
        producer,
        initializers,
    )

    injections: Dict[str, List[NodeProto]] = {}
    appended_initializers: List[TensorProto] = []
    for stage, layer_index, target_name in TARGET_SPECS:
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        new_nodes, new_initializers = make_qdq_nodes(
            model,
            stage,
            layer_index,
            target,
            producer,
            consumers,
            initializers,
            cache_scales,
            template,
        )
        injections[target_name] = new_nodes
        appended_initializers.extend(new_initializers)

    # 在各自 MatMul 之前插入 Q/DQ，保持 ONNX 拓扑顺序合法。
    rewritten_nodes: List[NodeProto] = []
    for node in model.graph.node:
        if node.name in injections:
            rewritten_nodes.extend(injections[node.name])
        rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(appended_initializers)

    validate_result(model, before_qdq, before_plugins)
    return before_qdq, before_plugins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V19：从稳定V16分叉，只为block1/2/3/4共28个Attention "
            "q/MatMul插入专用激活Q/DQ和axis=1权重Q/DQ"
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v16_"
            "block4_packed_dwconv_gelu.onnx"
        ),
        help="稳定的、尚未量化 Attention 的 V16 ONNX",
    )
    parser.add_argument(
        "--cache",
        default="deploy/egcienet_352_multiclass_int8.cache",
        help="TensorRT INT8 calibration cache",
    )
    parser.add_argument(
        "--output",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v19_"
            "all_blocks_attn_q_matmul_qdq.onnx"
        ),
        help="四个 stage 全部 Q 投影量化后的 ONNX",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已有输出",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    cache_path = Path(args.cache)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError("找不到稳定 V16 ONNX: {}".format(input_path))
    if not cache_path.is_file():
        raise FileNotFoundError("找不到 calibration cache: {}".format(cache_path))
    if input_path.resolve() == output_path.resolve():
        raise RuntimeError("输出不能覆盖 V16 输入 ONNX")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "输出已经存在: {}；如需覆盖请增加 --force".format(output_path)
        )

    print("[INPUT ] {}".format(input_path))
    print("[CACHE ] {}".format(cache_path))
    print("[OUTPUT] {}".format(output_path))
    print("[TARGET] block1=3, block2=4, block3=18, block4=3; total=28")
    print("[KEEP  ] SR/KV/MHA/projection/residual 和全部 Plugin 不变")

    cache_scales = v17.load_cache_scales(cache_path)
    model = onnx.load(str(input_path))
    before_qdq, before_plugins = rewrite_model(model, cache_scales)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    saved_model = onnx.load(str(output_path))
    validate_result(saved_model, before_qdq, before_plugins)

    print("\n[DONE] {}".format(output_path))
    print(
        "[NEXT] 构建后按 stage 核对28个 q/MatMul 是否全部从 "
        "f16f16 kernel 切换为 i8/s8 Tensor Core kernel"
    )


if __name__ == "__main__":
    main()
