#!/usr/bin/env python3

"""
V20：从稳定的 V16 ONNX 重新生成选择性 Attention INT8 图。

目标范围：
  - Q 投影：只量化 block3、block4，共 18 + 3 = 21 个；
  - KV 投影：量化 block1～block4，共 3 + 4 + 18 + 3 = 28 个；
  - block1、block2 的 Q 投影保持原始 FP16，不继承 V19 的 Q/DQ。

重要结构：
  - block1～3 的 KV 输入来自 SR 后的 attn/norm，各自使用专用激活 Q/DQ；
  - block4 没有独立 SR 输入，Q 和 KV 共用同一个 norm1 输出，因此每层只
    创建一套共享激活 Q/DQ，避免重复执行 Quantize；
  - Q/KV 静态权重分别使用 axis=1 的逐输出通道对称 INT8 Q/DQ；
  - SR、QK^T、Softmax、Attention×V、projection、残差和所有 Plugin 不修改。

本脚本直接读取 V16，而不是读取 V19 后删除节点。这样可以保证 block1/2 Q
路径中不会残留无消费者的 Q/DQ 或 scale initializer。
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Sequence, Tuple

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

Q_TARGET_SPECS: List[Tuple[str, int, int, str]] = [
    (
        "q",
        stage,
        layer_index,
        "/model/block{}.{}/attn/q/MatMul".format(stage, layer_index),
    )
    for stage in (3, 4)
    for layer_index in range(BLOCK_DEPTHS[stage])
]

KV_TARGET_SPECS: List[Tuple[str, int, int, str]] = [
    (
        "kv",
        stage,
        layer_index,
        "/model/block{}.{}/attn/kv/MatMul".format(stage, layer_index),
    )
    for stage, depth in BLOCK_DEPTHS.items()
    for layer_index in range(depth)
]

TARGET_SPECS = Q_TARGET_SPECS + KV_TARGET_SPECS
EXPECTED_QUANTIZED = {spec[3] for spec in TARGET_SPECS}


def create_activation_qdq(
    model: ModelProto,
    prefix: str,
    raw_activation: str,
    activation_scale: float,
    template: Mapping[str, object],
) -> Tuple[List[NodeProto], List[TensorProto], str]:
    """创建一套激活 per-tensor Q/DQ，并返回 DQ 输出名。"""
    scale_q_name = prefix + "_scale_1"
    scale_dq_name = prefix + "_scale_2"
    zero_q_name = prefix + "_zero_point_1"
    zero_dq_name = prefix + "_zero_point_2"
    q_output = prefix + "_QuantizeLinear_Output"
    dq_output = prefix + "_DequantizeLinear_Output"
    q_name = prefix + "_QuantizeLinear"
    dq_name = prefix + "_DequantizeLinear"

    v17.require_available_names(
        model,
        [q_name, dq_name],
        [
            scale_q_name,
            scale_dq_name,
            zero_q_name,
            zero_dq_name,
            q_output,
            dq_output,
        ],
    )

    scale_value = np.asarray(activation_scale).reshape(())
    zero_value = np.asarray(0, dtype=np.int8).reshape(())
    initializers = [
        v17.make_initializer_like(
            template["activation_scale"],
            scale_q_name,
            scale_value,
        ),
        v17.make_initializer_like(
            template["activation_scale"],
            scale_dq_name,
            scale_value,
        ),
        v17.make_initializer_like(
            template["activation_zero"],
            zero_q_name,
            zero_value,
        ),
        v17.make_initializer_like(
            template["activation_zero"],
            zero_dq_name,
            zero_value,
        ),
    ]

    q_node = helper.make_node(
        "QuantizeLinear",
        inputs=[raw_activation, scale_q_name, zero_q_name],
        outputs=[q_output],
        name=q_name,
    )
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=[q_output, scale_dq_name, zero_dq_name],
        outputs=[dq_output],
        name=dq_name,
    )
    return [q_node, dq_node], initializers, dq_output


def create_weight_qdq(
    model: ModelProto,
    prefix: str,
    raw_weight: str,
    weight_initializer: TensorProto,
    template: Mapping[str, object],
) -> Tuple[List[NodeProto], List[TensorProto], str]:
    """为一个 Q/KV 权重创建 axis=1 的逐输出通道 Q/DQ。"""
    weight = np.asarray(numpy_helper.to_array(weight_initializer))
    if weight.ndim != 2:
        raise RuntimeError("{} 不是二维 MatMul 权重: {}".format(raw_weight, weight.shape))

    # MatMul 权重布局为 [K,N]，axis=1 对应 N 个输出通道。
    weight_fp32 = weight.astype(np.float32, copy=False)
    weight_amax = np.max(np.abs(weight_fp32), axis=0)
    weight_scale = np.maximum(
        weight_amax / np.float32(127.0),
        np.float32(1.0e-8),
    )
    output_channels = int(weight.shape[1])
    if weight_scale.shape != (output_channels,):
        raise RuntimeError("{} 权重 scale shape 错误".format(raw_weight))
    if not np.all(np.isfinite(weight_scale)):
        raise RuntimeError("{} 权重 scale 存在非有限值".format(raw_weight))

    scale_q_name = prefix + "_scale_1"
    scale_dq_name = prefix + "_scale_2"
    zero_q_name = prefix + "_zero_point_1"
    zero_dq_name = prefix + "_zero_point_2"
    q_output = prefix + "_QuantizeLinear_Output"
    dq_output = prefix + "_DequantizeLinear_Output"
    q_name = prefix + "_QuantizeLinear"
    dq_name = prefix + "_DequantizeLinear"

    v17.require_available_names(
        model,
        [q_name, dq_name],
        [
            scale_q_name,
            scale_dq_name,
            zero_q_name,
            zero_dq_name,
            q_output,
            dq_output,
        ],
    )

    channel_zero = np.zeros((output_channels,), dtype=np.int8)
    initializers = [
        v17.make_initializer_like(
            template["weight_scale"],
            scale_q_name,
            weight_scale,
        ),
        v17.make_initializer_like(
            template["weight_scale"],
            scale_dq_name,
            weight_scale,
        ),
        v17.make_initializer_like(
            template["weight_zero"],
            zero_q_name,
            channel_zero,
        ),
        v17.make_initializer_like(
            template["weight_zero"],
            zero_dq_name,
            channel_zero,
        ),
    ]

    q_node = helper.make_node(
        "QuantizeLinear",
        inputs=[raw_weight, scale_q_name, zero_q_name],
        outputs=[q_output],
        name=q_name,
        axis=1,
    )
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=[q_output, scale_dq_name, zero_dq_name],
        outputs=[dq_output],
        name=dq_name,
        axis=1,
    )
    return [q_node, dq_node], initializers, dq_output


def require_clean_target(
    target: NodeProto,
    producer: Mapping[str, NodeProto],
    consumers: Mapping[str, Sequence[NodeProto]],
    initializers: Mapping[str, TensorProto],
) -> None:
    """确认目标仍是 V16 的原始 MatMul，且权重可以安全单独量化。"""
    if target.op_type != "MatMul" or len(target.input) != 2:
        raise RuntimeError("{} 不是合法 MatMul".format(target.name))

    activation_parent = producer.get(target.input[0])
    weight_parent = producer.get(target.input[1])
    if activation_parent is not None and activation_parent.op_type == "DequantizeLinear":
        raise RuntimeError("{} 激活已经量化".format(target.name))
    if weight_parent is not None and weight_parent.op_type == "DequantizeLinear":
        raise RuntimeError("{} 权重已经量化".format(target.name))

    if target.input[1] not in initializers:
        raise RuntimeError("{} 权重不是直接 initializer".format(target.name))
    weight_consumers = list(consumers.get(target.input[1], []))
    if len(weight_consumers) != 1 or weight_consumers[0].name != target.name:
        raise RuntimeError(
            "{} 权重不是专用权重: {}".format(
                target.name,
                [node.name for node in weight_consumers],
            )
        )


def validate_qdq_pair(
    target: NodeProto,
    producer: Mapping[str, NodeProto],
    initializers: Mapping[str, TensorProto],
) -> Tuple[NodeProto, NodeProto]:
    """验证一个 MatMul 的激活和权重都由完整 Q -> DQ 提供。"""
    activation_dq = producer.get(target.input[0])
    weight_dq = producer.get(target.input[1])
    if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
        raise RuntimeError("{} 激活输入不是直接 DQ".format(target.name))
    if weight_dq is None or weight_dq.op_type != "DequantizeLinear":
        raise RuntimeError("{} 权重输入不是直接 DQ".format(target.name))

    activation_q = producer.get(activation_dq.input[0])
    weight_q = producer.get(weight_dq.input[0])
    if activation_q is None or activation_q.op_type != "QuantizeLinear":
        raise RuntimeError("{} 激活缺少 Q -> DQ".format(target.name))
    if weight_q is None or weight_q.op_type != "QuantizeLinear":
        raise RuntimeError("{} 权重缺少 Q -> DQ".format(target.name))

    if v17.get_axis(weight_q) != 1 or v17.get_axis(weight_dq) != 1:
        raise RuntimeError("{} 权重 Q/DQ axis 不是1".format(target.name))

    weight = v17.initializer_array(initializers, weight_q.input[0])
    scale_q = v17.initializer_array(initializers, weight_q.input[1])
    scale_dq = v17.initializer_array(initializers, weight_dq.input[1])
    zero_q = v17.initializer_array(initializers, weight_q.input[2])
    zero_dq = v17.initializer_array(initializers, weight_dq.input[2])
    if scale_q.shape != (weight.shape[1],):
        raise RuntimeError("{} 权重 scale shape 错误".format(target.name))
    if scale_dq.shape != scale_q.shape or not np.array_equal(scale_q, scale_dq):
        raise RuntimeError("{} 权重 Q/DQ scale 不一致".format(target.name))
    if zero_q.dtype != np.int8 or zero_dq.dtype != np.int8:
        raise RuntimeError("{} 权重 zero-point 不是 INT8".format(target.name))
    if np.any(zero_q != 0) or np.any(zero_dq != 0):
        raise RuntimeError("{} 权重 zero-point 不是全零".format(target.name))

    activation_scale_q = v17.initializer_array(
        initializers,
        activation_q.input[1],
    )
    activation_scale_dq = v17.initializer_array(
        initializers,
        activation_dq.input[1],
    )
    if activation_scale_q.shape != () or activation_scale_dq.shape != ():
        raise RuntimeError("{} 激活 scale 不是标量".format(target.name))
    if not np.array_equal(activation_scale_q, activation_scale_dq):
        raise RuntimeError("{} 激活 Q/DQ scale 不一致".format(target.name))
    return activation_dq, weight_dq


def validate_result(
    model: ModelProto,
    before_qdq: Counter,
    before_plugins: Mapping[str, str],
) -> None:
    """验证 Q/KV 目标集合、block4 共享边界和全部非目标约束。"""
    onnx.checker.check_model(model)
    nodes_by_name, producer, consumers, initializers = v17.build_maps(model)

    quantized_attention_matmuls = {
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and "/attn/" in node.name
        and v17.is_quantized_weight_matmul(node, producer)
    }
    missing = sorted(EXPECTED_QUANTIZED - quantized_attention_matmuls)
    unexpected = sorted(quantized_attention_matmuls - EXPECTED_QUANTIZED)
    if missing or unexpected:
        raise RuntimeError(
            "Attention MatMul 量化集合错误:\n  missing={}\n  unexpected={}".format(
                missing,
                unexpected,
            )
        )

    # block1/2 Q 必须完全回到 V16 FP16 输入，不能残留 DQ。
    for stage in (1, 2):
        for layer_index in range(BLOCK_DEPTHS[stage]):
            q_name = "/model/block{}.{}/attn/q/MatMul".format(stage, layer_index)
            q_node = v17.require_node(nodes_by_name, q_name, "MatMul")
            if producer.get(q_node.input[0]) is not None and producer[q_node.input[0]].op_type == "DequantizeLinear":
                raise RuntimeError("{} 激活仍残留 Q/DQ".format(q_name))
            if v17.is_quantized_weight_matmul(q_node, producer):
                raise RuntimeError("{} 权重仍残留 Q/DQ".format(q_name))

    for _, _, _, target_name in TARGET_SPECS:
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        validate_qdq_pair(target, producer, initializers)

    # block4 的 Q 与 KV 必须逐层共用一套激活 DQ，且只供这两个 MatMul 使用。
    for layer_index in range(BLOCK_DEPTHS[4]):
        q_name = "/model/block4.{}/attn/q/MatMul".format(layer_index)
        kv_name = "/model/block4.{}/attn/kv/MatMul".format(layer_index)
        q_node = v17.require_node(nodes_by_name, q_name, "MatMul")
        kv_node = v17.require_node(nodes_by_name, kv_name, "MatMul")
        if q_node.input[0] != kv_node.input[0]:
            raise RuntimeError("block4.{} Q/KV 没有共享激活 DQ".format(layer_index))
        shared_consumers = {node.name for node in consumers.get(q_node.input[0], [])}
        if shared_consumers != {q_name, kv_name}:
            raise RuntimeError(
                "block4.{} 共享 DQ 消费者错误: {}".format(
                    layer_index,
                    sorted(shared_consumers),
                )
            )

    # 除 block4 的3套共享边界外，其他激活 DQ 都只能服务自己的 MatMul。
    block4_shared_outputs = {
        v17.require_node(
            nodes_by_name,
            "/model/block4.{}/attn/q/MatMul".format(layer_index),
            "MatMul",
        ).input[0]
        for layer_index in range(BLOCK_DEPTHS[4])
    }
    for _, _, _, target_name in TARGET_SPECS:
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        if target.input[0] in block4_shared_outputs:
            continue
        activation_consumers = consumers.get(target.input[0], [])
        if len(activation_consumers) != 1 or activation_consumers[0].name != target_name:
            raise RuntimeError(
                "{} 激活 DQ 不是专用分支: {}".format(
                    target_name,
                    [node.name for node in activation_consumers],
                )
            )

    # projection 继续保持高精度；其余 Attention MatMul 由精确集合检查覆盖。
    for stage, depth in BLOCK_DEPTHS.items():
        for layer_index in range(depth):
            proj_name = "/model/block{}.{}/attn/proj/MatMul".format(stage, layer_index)
            proj = v17.require_node(nodes_by_name, proj_name, "MatMul")
            if v17.is_quantized_weight_matmul(proj, producer):
                raise RuntimeError("出现非目标 projection 量化: {}".format(proj_name))

    # 49套权重边界 + 46套唯一激活边界 = 每种节点各新增95个。
    expected_added_each = 95
    after_qdq = v17.qdq_count(model)
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

    print("\n[VALIDATE]")
    print("  block1/2 q Q/DQ                 : 0/7")
    print("  block3/4 q INT8 targets         : 21/21")
    print("  block1～4 kv INT8 targets       : 28/28")
    print("  quantized Attention MatMul       : 49/49")
    print("  unique activation Q/DQ pairs     : 46")
    print("  block4 shared Q/KV activation    : 3/3")
    print("  weight Q/DQ pairs axis=1         : 49/49")
    print("  projection quantized             : 0")
    print("  added QuantizeLinear             : {}".format(expected_added_each))
    print("  added DequantizeLinear           : {}".format(expected_added_each))
    print("  egcinet Plugin nodes unchanged   : {}".format(len(after_plugins)))


def rewrite_model(
    model: ModelProto,
    cache_scales: Mapping[str, float],
) -> Tuple[Counter, Dict[str, str]]:
    """从干净 V16 插入目标 Q/KV Q/DQ，并自动共享相同激活边界。"""
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

    # 复用 block3.0 fc1 已验证边界的数据类型和 axis 格式，不复用数值 scale。
    template = v17.inspect_fc1_template(
        0,
        nodes_by_name,
        producer,
        initializers,
    )

    node_order = {node.name: index for index, node in enumerate(model.graph.node)}
    target_records: Dict[str, Dict[str, object]] = {}
    activation_groups: DefaultDict[str, List[str]] = defaultdict(list)

    # 步骤1：先保存 V16 原始输入，确认所有目标都可以安全修改。
    for kind, stage, layer_index, target_name in TARGET_SPECS:
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        require_clean_target(target, producer, consumers, initializers)
        target_records[target_name] = {
            "kind": kind,
            "stage": stage,
            "layer_index": layer_index,
            "node": target,
            "raw_activation": target.input[0],
            "raw_weight": target.input[1],
        }
        activation_groups[target.input[0]].append(target_name)

    if len(activation_groups) != 46:
        raise RuntimeError(
            "唯一激活输入数量不是46，实际为{}".format(len(activation_groups))
        )
    shared_groups = [names for names in activation_groups.values() if len(names) > 1]
    if len(shared_groups) != 3 or any(len(names) != 2 for names in shared_groups):
        raise RuntimeError("block4 Q/KV 共享关系异常: {}".format(shared_groups))

    injections: DefaultDict[str, List[NodeProto]] = defaultdict(list)
    appended_initializers: List[TensorProto] = []

    # 步骤2：按原始激活 tensor 分组；block4 Q/KV 会自然落入同一组。
    for group_index, (raw_activation, target_names) in enumerate(activation_groups.items()):
        first_target_name = min(target_names, key=lambda name: node_order[name])
        if len(target_names) == 2:
            first_record = target_records[first_target_name]
            prefix = "/model/block{}.{}/attn/V20_QKV_SharedActivation".format(
                first_record["stage"],
                first_record["layer_index"],
            )
        else:
            prefix = first_target_name + "_V20_Activation"

        cache_key, activation_scale = v17.find_activation_scale(
            raw_activation,
            producer,
            cache_scales,
        )
        new_nodes, new_initializers, dq_output = create_activation_qdq(
            model,
            prefix,
            raw_activation,
            activation_scale,
            template,
        )
        injections[first_target_name].extend(new_nodes)
        appended_initializers.extend(new_initializers)
        for target_name in target_names:
            target_records[target_name]["node"].input[0] = dq_output

        print(
            "[ACTIVATION] group={:02d}, consumers={}, scale={:.9g}, cache_key={}".format(
                group_index,
                target_names,
                activation_scale,
                cache_key,
            )
        )

    # 步骤3：每个 Q/KV MatMul 都有自己的静态权重，因此分别建立权重 Q/DQ。
    for kind, stage, layer_index, target_name in TARGET_SPECS:
        record = target_records[target_name]
        raw_weight = record["raw_weight"]
        weight_initializer = initializers[raw_weight]
        prefix = target_name + "_V20_Weight"
        new_nodes, new_initializers, dq_output = create_weight_qdq(
            model,
            prefix,
            raw_weight,
            weight_initializer,
            template,
        )
        injections[target_name].extend(new_nodes)
        appended_initializers.extend(new_initializers)
        record["node"].input[1] = dq_output

        weight_shape = tuple(numpy_helper.to_array(weight_initializer).shape)
        print(
            "[WEIGHT] block{}.{}, kind={}, shape={}".format(
                stage,
                layer_index,
                kind,
                weight_shape,
            )
        )

    # 步骤4：将新节点放到对应 MatMul 前面，保持 ONNX 拓扑顺序合法。
    rewritten_nodes: List[NodeProto] = []
    for node in model.graph.node:
        rewritten_nodes.extend(injections.get(node.name, []))
        rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(appended_initializers)

    validate_result(model, before_qdq, before_plugins)
    return before_qdq, before_plugins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V20：从稳定V16生成Q仅block3/4、KV覆盖block1～4的显式Q/DQ图"
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
            "deploy/egcienet_352_multiclass_qdq_v20_"
            "q_block34_kv_all.onnx"
        ),
        help="选择性 Q 与全 stage KV 量化后的 ONNX",
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
    print("[Q     ] block1/2=FP16, block3/4=INT8; total=21")
    print("[KV    ] block1/2/3/4=INT8; total=28")
    print("[SHARE ] block4 每层 Q/KV 共用一套激活 Q/DQ")

    cache_scales = v17.load_cache_scales(cache_path)
    model = onnx.load(str(input_path))
    before_qdq, before_plugins = rewrite_model(model, cache_scales)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    saved_model = onnx.load(str(output_path))
    validate_result(saved_model, before_qdq, before_plugins)

    print("\n[DONE] {}".format(output_path))
    print(
        "[NEXT] 构建 engine 后分别统计 block3/4 Q 与 block1～4 KV 的 "
        "GEMM、Quantize 和完整链耗时"
    )


if __name__ == "__main__":
    main()
