#!/usr/bin/env python3

"""
V21：根据 V20 分 stage 测试结果，删除负收益的 block1 KV Q/DQ。

本版继续从稳定 V16 直接生成目标图，不在 V20 ONNX 上删除节点：
  - block1：Q、KV 都保持 FP16；
  - block2：Q 保持 FP16，只量化 KV；
  - block3：量化 Q 和 KV；
  - block4：量化 Q 和 KV，并让二者共用同一套激活 Q/DQ；
  - projection、SR、QK^T、Softmax、Attention×V、残差和 Plugin 不修改。

低层 Q/DQ 创建函数复用 V20 已经验证过的实现；V21 自己定义目标集合、
图重写和严格验证，V20 文件本身保持不变，便于性能版本对照。
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Sequence, Tuple

import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

import insert_attn_q_block34_kv_all_qdq_v20 as v20


v17 = v20.v17
BLOCK_DEPTHS: Dict[int, int] = dict(v20.BLOCK_DEPTHS)

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
    for stage in (2, 3, 4)
    for layer_index in range(BLOCK_DEPTHS[stage])
]

TARGET_SPECS = Q_TARGET_SPECS + KV_TARGET_SPECS
EXPECTED_QUANTIZED = {spec[3] for spec in TARGET_SPECS}


def require_unquantized_matmul(
    nodes_by_name: Mapping[str, NodeProto],
    producer: Mapping[str, NodeProto],
    node_name: str,
) -> None:
    """确认指定 MatMul 的激活和权重都没有残留 Attention Q/DQ。"""
    node = v17.require_node(nodes_by_name, node_name, "MatMul")
    activation_parent = producer.get(node.input[0])
    if activation_parent is not None and activation_parent.op_type == "DequantizeLinear":
        raise RuntimeError("{} 激活仍残留 Q/DQ".format(node_name))
    if v17.is_quantized_weight_matmul(node, producer):
        raise RuntimeError("{} 权重仍残留 Q/DQ".format(node_name))


def validate_result(
    model: ModelProto,
    before_qdq: Counter,
    before_plugins: Mapping[str, str],
) -> None:
    """验证 V21 的目标集合、共享边界和非目标高精度约束。"""
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

    # block1 的 Q/KV 全部回退，block2 的 Q 继续保持 FP16。
    for layer_index in range(BLOCK_DEPTHS[1]):
        require_unquantized_matmul(
            nodes_by_name,
            producer,
            "/model/block1.{}/attn/q/MatMul".format(layer_index),
        )
        require_unquantized_matmul(
            nodes_by_name,
            producer,
            "/model/block1.{}/attn/kv/MatMul".format(layer_index),
        )
    for layer_index in range(BLOCK_DEPTHS[2]):
        require_unquantized_matmul(
            nodes_by_name,
            producer,
            "/model/block2.{}/attn/q/MatMul".format(layer_index),
        )

    # 46个目标必须同时具有激活 Q/DQ 和 axis=1 权重 Q/DQ。
    for _, _, _, target_name in TARGET_SPECS:
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        v20.validate_qdq_pair(target, producer, initializers)

    # block4 的 Q/KV 共用 norm1 激活，所以每层只允许一套共享 DQ。
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

    # 其他激活 DQ 必须是单目标分支，避免量化边界污染旁路。
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

    # projection 必须保持高精度；精确的 MatMul 集合检查会阻止其他 Attention 误量化。
    for stage, depth in BLOCK_DEPTHS.items():
        for layer_index in range(depth):
            proj_name = "/model/block{}.{}/attn/proj/MatMul".format(stage, layer_index)
            proj = v17.require_node(nodes_by_name, proj_name, "MatMul")
            if v17.is_quantized_weight_matmul(proj, producer):
                raise RuntimeError("出现非目标 projection 量化: {}".format(proj_name))

    # 46套权重边界 + 43套唯一激活边界 = 每种节点各新增89个。
    expected_added_each = 89
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
    print("  block1 q/kv Q/DQ               : 0/6")
    print("  block2 q Q/DQ                  : 0/4")
    print("  block3/4 q INT8 targets        : 21/21")
    print("  block2/3/4 kv INT8 targets     : 25/25")
    print("  quantized Attention MatMul      : 46/46")
    print("  unique activation Q/DQ pairs    : 43")
    print("  block4 shared Q/KV activation   : 3/3")
    print("  weight Q/DQ pairs axis=1        : 46/46")
    print("  projection quantized            : 0")
    print("  added QuantizeLinear            : {}".format(expected_added_each))
    print("  added DequantizeLinear          : {}".format(expected_added_each))
    print("  egcinet Plugin nodes unchanged  : {}".format(len(after_plugins)))


def rewrite_model(
    model: ModelProto,
    cache_scales: Mapping[str, float],
) -> Tuple[Counter, Dict[str, str]]:
    """从干净 V16 插入 V21 目标 Q/KV Q/DQ。"""
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

    # 只借用成功 fc1 边界的数据类型和 axis 格式，各目标数值 scale 独立计算。
    template = v17.inspect_fc1_template(
        0,
        nodes_by_name,
        producer,
        initializers,
    )

    node_order = {node.name: index for index, node in enumerate(model.graph.node)}
    target_records: Dict[str, Dict[str, object]] = {}
    activation_groups: DefaultDict[str, List[str]] = defaultdict(list)

    # 步骤1：记录 V16 的原始输入，并按激活 tensor 建立共享分组。
    for kind, stage, layer_index, target_name in TARGET_SPECS:
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        v20.require_clean_target(target, producer, consumers, initializers)
        target_records[target_name] = {
            "kind": kind,
            "stage": stage,
            "layer_index": layer_index,
            "node": target,
            "raw_activation": target.input[0],
            "raw_weight": target.input[1],
        }
        activation_groups[target.input[0]].append(target_name)

    if len(activation_groups) != 43:
        raise RuntimeError(
            "唯一激活输入数量不是43，实际为{}".format(len(activation_groups))
        )
    shared_groups = [names for names in activation_groups.values() if len(names) > 1]
    if len(shared_groups) != 3 or any(len(names) != 2 for names in shared_groups):
        raise RuntimeError("block4 Q/KV 共享关系异常: {}".format(shared_groups))

    injections: DefaultDict[str, List[NodeProto]] = defaultdict(list)
    appended_initializers: List[TensorProto] = []

    # 步骤2：创建43套唯一激活边界，block4 的 Q/KV 自动共享。
    for group_index, (raw_activation, target_names) in enumerate(activation_groups.items()):
        first_target_name = min(target_names, key=lambda name: node_order[name])
        if len(target_names) == 2:
            first_record = target_records[first_target_name]
            prefix = "/model/block{}.{}/attn/V21_QKV_SharedActivation".format(
                first_record["stage"],
                first_record["layer_index"],
            )
        else:
            prefix = first_target_name + "_V21_Activation"

        cache_key, activation_scale = v17.find_activation_scale(
            raw_activation,
            producer,
            cache_scales,
        )
        new_nodes, new_initializers, dq_output = v20.create_activation_qdq(
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

    # 步骤3：为21个 Q 和25个 KV 分别创建逐输出通道权重 Q/DQ。
    for kind, stage, layer_index, target_name in TARGET_SPECS:
        record = target_records[target_name]
        raw_weight = record["raw_weight"]
        weight_initializer = initializers[raw_weight]
        prefix = target_name + "_V21_Weight"
        new_nodes, new_initializers, dq_output = v20.create_weight_qdq(
            model,
            prefix,
            raw_weight,
            weight_initializer,
            template,
        )
        injections[target_name].extend(new_nodes)
        appended_initializers.extend(new_initializers)
        record["node"].input[1] = dq_output

        print(
            "[WEIGHT] block{}.{}, kind={}, shape={}".format(
                stage,
                layer_index,
                kind,
                tuple(numpy_helper.to_array(weight_initializer).shape),
            )
        )

    # 步骤4：把新边界放到各自 MatMul 前，保持 ONNX 拓扑顺序合法。
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
            "V21：从稳定V16生成Q=block3/4、KV=block2/3/4的显式Q/DQ图"
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
            "deploy/egcienet_352_multiclass_qdq_v21_"
            "q_block34_kv_block234.onnx"
        ),
        help="移除 block1 KV Q/DQ 后的 ONNX",
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
    print("[KV    ] block1=FP16, block2/3/4=INT8; total=25")
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
        "[NEXT] 构建后确认 block1 KV 回到 FP16，并复测 Q/KV 完整链与整网耗时"
    )


if __name__ == "__main__":
    main()
