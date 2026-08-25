#!/usr/bin/env python3

"""
V23：以 V22 的 Q/KV 布局为基线，只量化 block1 的三个 QK^T MatMul。

最终目标：
  - 保留 V22：Q=block1/3/4，KV=block1/2/3/4；
  - 新增 block1.0～2 的 /attn/MatMul 两侧激活 Q/DQ；
  - QK 输出继续为 FP16，后面的缩放 Mul、Softmax、Attention×V 和
    projection 均不量化；
  - 不修改任何 egcinet Plugin。

QK 没有静态权重，两侧都是动态激活，所以每层分别需要 Q 与 K^T 两套
per-tensor Q/DQ，三个 block1 层合计新增6套。
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto

import insert_attn_q_block134_kv_all_qdq_v22 as v22


v20 = v22.v20
v17 = v22.v17
BLOCK_DEPTHS: Dict[int, int] = dict(v22.BLOCK_DEPTHS)

BLOCK1_QK_TARGETS: List[str] = [
    "/model/block1.{}/attn/MatMul".format(layer_index)
    for layer_index in range(BLOCK_DEPTHS[1])
]


def find_qk_input_scale(
    tensor_name: str,
    producer: Mapping[str, NodeProto],
    cache_scales: Mapping[str, float],
) -> Tuple[str, float, str]:
    """
    为 QK 输入查找 scale。

    优先使用 QK 直接输入 tensor 的精确统计；若 cache 没记录中间布局节点，
    只允许沿 Transpose/Reshape/Identity/Gather 的数据输入向上追溯。前3种
    算子不改变数值集合；Gather 只取子集，使用其上游 scale 不会增加溢出，
    但可能略保守。不会跨越 Add、MatMul 或非布局算子。
    """
    allowed_passthrough = {"Transpose", "Reshape", "Identity", "Gather"}
    visited = set()
    current = tensor_name

    while current and current not in visited:
        visited.add(current)
        for key in (current + "_scale", current):
            if key not in cache_scales:
                continue
            scale = float(cache_scales[key])
            if not np.isfinite(scale) or scale <= 0.0:
                raise RuntimeError("calibration scale 非法: {}={}".format(key, scale))
            return key, scale, current

        parent = producer.get(current)
        if (
            parent is None
            or parent.op_type not in allowed_passthrough
            or not parent.input
        ):
            break
        current = parent.input[0]

    nearby = sorted(
        key
        for key in cache_scales
        if tensor_name in key or "/block1." in key and "/attn/" in key
    )[:30]
    raise RuntimeError(
        "找不到 QK 输入 calibration scale:\n"
        "  tensor  : {}\n"
        "  visited : {}\n"
        "  nearby  : {}".format(tensor_name, sorted(visited), nearby)
    )


def validate_activation_input(
    target: NodeProto,
    input_index: int,
    producer: Mapping[str, NodeProto],
    consumers: Mapping[str, Sequence[NodeProto]],
    initializers: Mapping[str, TensorProto],
) -> None:
    """验证 QK 的一个输入是完整、专用的 per-tensor Q -> DQ。"""
    dq_node = producer.get(target.input[input_index])
    if dq_node is None or dq_node.op_type != "DequantizeLinear":
        raise RuntimeError(
            "{} input{} 不是直接 DQ".format(target.name, input_index)
        )
    q_node = producer.get(dq_node.input[0])
    if q_node is None or q_node.op_type != "QuantizeLinear":
        raise RuntimeError(
            "{} input{} 缺少 Q -> DQ".format(target.name, input_index)
        )

    dq_consumers = consumers.get(dq_node.output[0], [])
    if len(dq_consumers) != 1 or dq_consumers[0].name != target.name:
        raise RuntimeError(
            "{} input{} DQ 不是专用边界: {}".format(
                target.name,
                input_index,
                [node.name for node in dq_consumers],
            )
        )

    scale_q = v17.initializer_array(initializers, q_node.input[1])
    scale_dq = v17.initializer_array(initializers, dq_node.input[1])
    zero_q = v17.initializer_array(initializers, q_node.input[2])
    zero_dq = v17.initializer_array(initializers, dq_node.input[2])
    if scale_q.shape != () or scale_dq.shape != ():
        raise RuntimeError(
            "{} input{} scale 不是标量".format(target.name, input_index)
        )
    if not np.array_equal(scale_q, scale_dq):
        raise RuntimeError(
            "{} input{} Q/DQ scale 不一致".format(target.name, input_index)
        )
    if zero_q.dtype != np.int8 or zero_dq.dtype != np.int8:
        raise RuntimeError(
            "{} input{} zero-point 不是 INT8".format(target.name, input_index)
        )
    if np.any(zero_q != 0) or np.any(zero_dq != 0):
        raise RuntimeError(
            "{} input{} zero-point 不是0".format(target.name, input_index)
        )


def validate_result(
    model: ModelProto,
    before_qdq: Counter,
    before_plugins: Mapping[str, str],
) -> None:
    """验证 V22 投影基线和 block1 三个 QK 边界均准确保留。"""
    onnx.checker.check_model(model)
    nodes_by_name, producer, consumers, initializers = v17.build_maps(model)

    # 带静态权重的 Attention 投影集合必须仍与 V22 完全一致。
    # 旧辅助函数通过 input1 的 Q -> DQ 判断“权重量化”；QK 的 input1 是
    # 动态 K^T，量化后也满足这个结构。因此这里只在真正的 q/kv/proj
    # 投影名称范围内统计，不能把 QK 当成静态权重 MatMul。
    projection_suffixes = ("/q/MatMul", "/kv/MatMul", "/proj/MatMul")
    quantized_weight_matmuls = {
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and "/attn/" in node.name
        and node.name.endswith(projection_suffixes)
        and v17.is_quantized_weight_matmul(node, producer)
    }
    missing = sorted(v22.EXPECTED_QUANTIZED - quantized_weight_matmuls)
    unexpected = sorted(quantized_weight_matmuls - v22.EXPECTED_QUANTIZED)
    if missing or unexpected:
        raise RuntimeError(
            "V22 Q/KV 量化集合发生变化:\n  missing={}\n  unexpected={}".format(
                missing,
                unexpected,
            )
        )

    # 52个 Q/KV 投影的输入 Q/DQ 继续完整有效。
    for _, _, _, target_name in v22.TARGET_SPECS:
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        v20.validate_qdq_pair(target, producer, initializers)

    # block2 Q 是 V22 唯一保持 FP16 的 Q 投影。
    for layer_index in range(BLOCK_DEPTHS[2]):
        q_name = "/model/block2.{}/attn/q/MatMul".format(layer_index)
        q_node = v17.require_node(nodes_by_name, q_name, "MatMul")
        activation_parent = producer.get(q_node.input[0])
        if activation_parent is not None and activation_parent.op_type == "DequantizeLinear":
            raise RuntimeError("{} 激活错误地量化".format(q_name))
        if v17.is_quantized_weight_matmul(q_node, producer):
            raise RuntimeError("{} 权重错误地量化".format(q_name))

    # block4 Q/KV 必须继续共享 V22 的3套激活边界。
    for layer_index in range(BLOCK_DEPTHS[4]):
        q_name = "/model/block4.{}/attn/q/MatMul".format(layer_index)
        kv_name = "/model/block4.{}/attn/kv/MatMul".format(layer_index)
        q_node = v17.require_node(nodes_by_name, q_name, "MatMul")
        kv_node = v17.require_node(nodes_by_name, kv_name, "MatMul")
        if q_node.input[0] != kv_node.input[0]:
            raise RuntimeError("block4.{} Q/KV 共享被破坏".format(layer_index))

    # 只有 block1 的三个 QK MatMul 两侧应该出现激活 Q/DQ。
    for stage, depth in BLOCK_DEPTHS.items():
        for layer_index in range(depth):
            qk_name = "/model/block{}.{}/attn/MatMul".format(stage, layer_index)
            qk = v17.require_node(nodes_by_name, qk_name, "MatMul")
            if stage == 1:
                validate_activation_input(qk, 0, producer, consumers, initializers)
                validate_activation_input(qk, 1, producer, consumers, initializers)
            else:
                for input_index in (0, 1):
                    parent = producer.get(qk.input[input_index])
                    if parent is not None and parent.op_type == "DequantizeLinear":
                        raise RuntimeError(
                            "出现非目标 QK 输入量化: {} input{}".format(
                                qk_name,
                                input_index,
                            )
                        )

            # QK 输出必须直接进入原始缩放 Mul，不能继续延伸 INT8 边界。
            qk_consumers = consumers.get(qk.output[0], [])
            if len(qk_consumers) != 1 or qk_consumers[0].op_type != "Mul":
                raise RuntimeError(
                    "{} 输出链发生变化: {}".format(
                        qk_name,
                        [(node.name, node.op_type) for node in qk_consumers],
                    )
                )

            # Attention×V 继续保持 Softmax/Gather 高精度输入。
            av_name = "/model/block{}.{}/attn/MatMul_1".format(stage, layer_index)
            av = v17.require_node(nodes_by_name, av_name, "MatMul")
            for input_index in (0, 1):
                parent = producer.get(av.input[input_index])
                if parent is not None and parent.op_type == "DequantizeLinear":
                    raise RuntimeError(
                        "出现非目标 Attention×V 输入量化: {} input{}".format(
                            av_name,
                            input_index,
                        )
                    )

            proj_name = "/model/block{}.{}/attn/proj/MatMul".format(stage, layer_index)
            proj = v17.require_node(nodes_by_name, proj_name, "MatMul")
            if v17.is_quantized_weight_matmul(proj, producer):
                raise RuntimeError("projection 被意外量化: {}".format(proj_name))

    # V22 每种节点新增101个，V23的6套QK输入再增加6个，总计107个。
    expected_added_each = 107
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
        raise RuntimeError("egcinet Plugin 节点发生变化")

    print("\n[VALIDATE V23]")
    print("  V22 quantized Q/KV projections : 52/52")
    print("  block1 QK INT8 targets          : 3/3")
    print("  block1 QK quantized inputs       : 6/6")
    print("  block2/3/4 QK quantized          : 0")
    print("  Softmax/Attention×V/Proj changed : 0")
    print("  added QuantizeLinear             : {}".format(expected_added_each))
    print("  added DequantizeLinear           : {}".format(expected_added_each))
    print("  egcinet Plugin nodes unchanged   : {}".format(len(after_plugins)))


def rewrite_model(
    model: ModelProto,
    cache_scales: Mapping[str, float],
) -> Tuple[Counter, Dict[str, str]]:
    """先构建 V22，再为 block1 三个 QK 插入两侧激活 Q/DQ。"""
    onnx.checker.check_model(model)
    before_qdq = v17.qdq_count(model)
    before_plugins = v17.plugin_snapshot(model)

    # 步骤1：直接复用 V22 的完整重写和严格验证，得到确定的 Q/KV 基线。
    v22.rewrite_model(model, cache_scales)
    nodes_by_name, producer, consumers, initializers = v17.build_maps(model)
    template = v17.inspect_fc1_template(
        0,
        nodes_by_name,
        producer,
        initializers,
    )

    injections: DefaultDict[str, List[NodeProto]] = defaultdict(list)
    appended_initializers: List[TensorProto] = []

    # 步骤2：三个 QK MatMul 各自量化 Q 和 K^T 两个动态输入。
    for layer_index, target_name in enumerate(BLOCK1_QK_TARGETS):
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        if len(target.input) != 2:
            raise RuntimeError("{} 输入数量不是2".format(target_name))

        expected_parent_ops = ("Transpose", "Transpose")
        for input_index, side in ((0, "Q"), (1, "K")):
            raw_input = target.input[input_index]
            parent = producer.get(raw_input)
            if parent is None or parent.op_type != expected_parent_ops[input_index]:
                raise RuntimeError(
                    "{} {} 输入父节点不是 Transpose: {}".format(
                        target_name,
                        side,
                        None if parent is None else (parent.name, parent.op_type),
                    )
                )
            raw_consumers = consumers.get(raw_input, [])
            if len(raw_consumers) != 1 or raw_consumers[0].name != target_name:
                raise RuntimeError(
                    "{} {} 输入不是 QK 专用 tensor: {}".format(
                        target_name,
                        side,
                        [node.name for node in raw_consumers],
                    )
                )

            cache_key, activation_scale, scale_tensor = find_qk_input_scale(
                raw_input,
                producer,
                cache_scales,
            )
            prefix = target_name + "_V23_{}_Input".format(side)
            new_nodes, new_initializers, dq_output = v20.create_activation_qdq(
                model,
                prefix,
                raw_input,
                activation_scale,
                template,
            )
            injections[target_name].extend(new_nodes)
            appended_initializers.extend(new_initializers)
            target.input[input_index] = dq_output

            print(
                "[QK INPUT] block1.{}, side={}, scale={:.9g}, "
                "cache_key={}, scale_tensor={}".format(
                    layer_index,
                    side,
                    activation_scale,
                    cache_key,
                    scale_tensor,
                )
            )

    # 步骤3：将6套 Q/DQ 放在各自 QK MatMul 前，保持拓扑顺序合法。
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
        description="V23：V22 Q/KV 基线上量化 block1 的三个 QK MatMul"
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
            "deploy/egcienet_352_multiclass_qdq_v23_"
            "q_block134_kv_all_block1_qk.onnx"
        ),
        help="新增 block1 三个 QK Q/DQ 后的 ONNX",
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
    print("[BASE  ] V22: Q=block1/3/4, KV=block1/2/3/4")
    print("[QK    ] block1.0～2，共3个 MatMul、6个动态 INT8 输入")
    print("[KEEP  ] QK输出、Mul、Softmax、Attention×V、Proj保持FP16")

    cache_scales = v17.load_cache_scales(cache_path)
    model = onnx.load(str(input_path))
    before_qdq, before_plugins = rewrite_model(model, cache_scales)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    saved_model = onnx.load(str(output_path))
    validate_result(saved_model, before_qdq, before_plugins)

    print("\n[DONE] {}".format(output_path))
    print(
        "[NEXT] 构建后确认3个 /attn/MatMul 的两个输入均为 Int8、输出为 Half，"
        "并分别统计两侧 Quantize、QK GEMM 和完整 QK 链耗时"
    )


if __name__ == "__main__":
    main()
