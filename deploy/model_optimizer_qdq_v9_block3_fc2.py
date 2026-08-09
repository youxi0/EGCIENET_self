#!/usr/bin/env python3

"""
V9：以 V7 为基线，仅额外量化 encoder block3 的 18 个 fc2/MatMul。

保留：
  - V5 的 12 个 FEM Conv + 3 个 GAM conv_pre Conv；
  - d31、d42、d42_31 三条 Resize 显式 INT8 Q/DQ 岛；
  - block1.0~2 的 3 个 fc1/MatMul；
  - block3.0~17 的 18 个 fc1/MatMul。

新增：
  - /model/block3.0~17/mlp/fc2/MatMul，共 18 个。

明确不量化：
  - V8 新增的 block1 DWConv；
  - block1、block2、block4 的 fc2；
  - block1/2/3/4 DWConv；
  - attention MatMul、LayerNorm、GELU 和其他 encoder 算子。

fc2 只量化激活输入和权重，输出保持高精度，供后续残差 Add 使用。
"""

import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Iterable, List, Set

import onnx
from onnx import ModelProto
from modelopt.onnx.quantization import quantize
from modelopt.onnx.quantization.calib_utils import import_scales_from_calib_cache

import model_optimizer_qdq_v7_block1_fc1 as v7


v6 = v7.v6
v5 = v7.v5

BLOCK3_FC2_TARGETS: List[str] = [
    "/model/block3.{}/mlp/fc2/MatMul".format(index)
    for index in range(18)
]

ALL_MATMUL_TARGETS: List[str] = [
    *v7.ALL_FC1_TARGETS,
    *BLOCK3_FC2_TARGETS,
]

ALL_WEIGHTED_TARGETS: List[str] = [
    *v5.TARGET_CONVS,
    *ALL_MATMUL_TARGETS,
]


def isolate_matmul_activation_qdq(
    model: ModelProto,
    target_names: Iterable[str],
) -> None:
    """保证每个目标 MatMul 的激活 DQ 只服务于该目标 MatMul。"""
    targets = list(target_names)
    target_set = set(targets)
    nodes_by_name, producer, consumers = v5.build_node_maps(model)

    print("\n[INFO] 隔离 {} 个目标 MatMul 的激活 Q/DQ".format(len(targets)))

    for target_name in targets:
        matmul = nodes_by_name[target_name]
        if matmul.op_type != "MatMul" or len(matmul.input) < 2:
            raise RuntimeError("{} 不是合法 MatMul".format(target_name))

        activation_dq = producer.get(matmul.input[0])
        if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
            raise RuntimeError(
                "{} 的激活输入不是直接 DQ: {}".format(
                    target_name,
                    matmul.input[0],
                )
            )

        activation_q = producer.get(activation_dq.input[0])
        if activation_q is None or activation_q.op_type != "QuantizeLinear":
            raise RuntimeError("{} 前面没有完整 Q -> DQ".format(target_name))

        raw_activation = activation_q.input[0]
        dq_output = activation_dq.output[0]
        dq_consumers = list(consumers.get(dq_output, []))
        target_consumers = [
            node for node in dq_consumers if node.name in target_set
        ]

        if len(target_consumers) != 1 or target_consumers[0].name != target_name:
            raise RuntimeError(
                "{} 的 DQ 目标消费者异常: {}".format(
                    target_name,
                    [node.name for node in target_consumers],
                )
            )

        rewired = []
        for consumer in dq_consumers:
            if consumer.name == target_name:
                continue

            for input_index, input_name in enumerate(consumer.input):
                if input_name == dq_output:
                    consumer.input[input_index] = raw_activation
                    rewired.append(
                        "{}[input {}]".format(
                            consumer.name or "<unnamed>",
                            input_index,
                        )
                    )

        if rewired:
            print("[REWIRE] {} bypass: {}".format(target_name, ", ".join(rewired)))
        else:
            print("[KEEP] {} 的激活 DQ 已专用".format(target_name))


def validate_matmul_qdq(model: ModelProto) -> None:
    """严格检查仅有 V7 fc1 和 block3 fc2 共 39 个 MatMul 被量化。"""
    nodes_by_name, producer, consumers = v5.build_node_maps(model)
    expected: Set[str] = set(ALL_MATMUL_TARGETS)

    weight_quantized_matmuls: Set[str] = {
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and v6.is_weight_quantized(node, producer)
    }

    missing = sorted(expected - weight_quantized_matmuls)
    unexpected = sorted(weight_quantized_matmuls - expected)
    activation_dq_errors = []
    fc2_output_q_errors = []

    for target_name in ALL_MATMUL_TARGETS:
        matmul = nodes_by_name[target_name]
        activation_dq = producer.get(matmul.input[0])

        if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
            activation_dq_errors.append(
                "{}: no activation DQ".format(target_name)
            )
            continue

        dq_consumers = consumers.get(activation_dq.output[0], [])
        if len(dq_consumers) != 1 or dq_consumers[0].name != target_name:
            activation_dq_errors.append(
                "{}: {}".format(
                    target_name,
                    [node.name for node in dq_consumers],
                )
            )

    # fc2 后面是残差 Add；不允许把 fc2 输出继续量化成新的 INT8 岛。
    for target_name in BLOCK3_FC2_TARGETS:
        matmul = nodes_by_name[target_name]
        for output_name in matmul.output:
            output_quantizers = [
                node.name or "<unnamed QuantizeLinear>"
                for node in consumers.get(output_name, [])
                if node.op_type == "QuantizeLinear"
            ]
            if output_quantizers:
                fc2_output_q_errors.append(
                    "{}: {}".format(target_name, output_quantizers)
                )

    if missing:
        raise RuntimeError("目标 MatMul 未量化: {}".format(missing))
    if unexpected:
        raise RuntimeError(
            "出现非目标 MatMul 权重量化（可能污染 fc2/attention）: {}".format(
                unexpected
            )
        )
    if activation_dq_errors:
        raise RuntimeError(
            "目标 MatMul 的激活 DQ 非专用: {}".format(activation_dq_errors)
        )
    if fc2_output_q_errors:
        raise RuntimeError(
            "block3 fc2 输出被继续量化，无法保持高精度残差边界: {}".format(
                fc2_output_q_errors
            )
        )


def validate_quantized_model(model: ModelProto) -> None:
    onnx.checker.check_model(model)

    # 保证仍然只有原有 15 个 Decoder Conv 被量化；任何 DWConv 进入
    # 权重量化集合都会作为 unexpected Conv 直接报错。
    v5.validate_target_conv_qdq(model)
    validate_matmul_qdq(model)

    for module_name, (resize_name, conv_name) in v5.GAM_RESIZE_TARGETS.items():
        v5.validate_gam_resize_island(
            model,
            module_name,
            resize_name,
            conv_name,
        )

    qdq_count = Counter(
        node.op_type
        for node in model.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
    )
    print("  Q/DQ nodes          : {}".format(dict(qdq_count)))
    print("  INT8 target Conv    : 15/15")
    print("  INT8 block1 fc1     : 3/3")
    print("  INT8 block3 fc1     : 18/18")
    print("  INT8 block3 fc2     : 18/18")
    print("  INT8 non-target MM  : 0")
    print("  INT8 encoder DWConv : 0")
    print("\n[PASS] V9 仅新增 block3 fc2；fc2 输出保持高精度边界")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "以 V7 为基线，仅额外量化 block3 的 18 个 fc2/MatMul；"
            "不继承 V8 的 block1 DWConv Q/DQ。"
        )
    )
    parser.add_argument(
        "--input",
        default="deploy/egcienet_352_multiclass.onnx",
        help="原始 ONNX 模型",
    )
    parser.add_argument(
        "--cache",
        default="deploy/egcienet_352_multiclass_int8.cache",
        help="TensorRT INT8 calibration cache",
    )
    parser.add_argument(
        "--output",
        default="deploy/egcienet_352_multiclass_qdq_v9_block3_fc2.onnx",
        help="最终显式 Q/DQ ONNX",
    )
    parser.add_argument(
        "--high-precision-dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="非 Q/DQ 区域的 ModelOpt 高精度类型；默认 fp16",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已有输出文件",
    )
    parser.add_argument(
        "--keep-temporary",
        action="store_true",
        help="保留未经 GAM Resize 传播的 ModelOpt 中间模型",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    cache_path = Path(args.cache)
    output_path = Path(args.output)
    temporary_path = output_path.with_name(
        "{}.modelopt_tmp.onnx".format(output_path.stem)
    )

    if not input_path.is_file():
        raise FileNotFoundError("找不到输入模型: {}".format(input_path))
    if not cache_path.is_file():
        raise FileNotFoundError("找不到 calibration cache: {}".format(cache_path))
    if input_path.resolve() == output_path.resolve():
        raise RuntimeError("输出路径不能覆盖原始 ONNX")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "输出已经存在: {}；如需覆盖请增加 --force".format(output_path)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary_path.exists():
        temporary_path.unlink()

    print("[ENV   ] ModelOpt {}".format(version("nvidia-modelopt")))
    print("[INPUT ] {}".format(input_path))
    print("[CACHE ] {}".format(cache_path))
    print("[TEMP  ] {}".format(temporary_path))
    print("[OUTPUT] {}".format(output_path))
    print("[HP    ] {}".format(args.high_precision_dtype))
    print(
        "[TARGET] 15 Decoder Conv + 3 block1 fc1 + "
        "18 block3 fc1 + 18 block3 fc2"
    )
    print("[BASE  ] V7（不包含 V8 block1 DWConv Q/DQ）")

    source_model = onnx.load(str(input_path))
    onnx.checker.check_model(source_model)
    v5.require_nodes(source_model, v5.TARGET_CONVS, "目标 Decoder Conv")
    v5.require_nodes(
        source_model,
        v7.BLOCK1_FC1_TARGETS,
        "目标 block1 fc1 MatMul",
    )
    v5.require_nodes(
        source_model,
        v6.BLOCK3_FC1_TARGETS,
        "目标 block3 fc1 MatMul",
    )
    v5.require_nodes(
        source_model,
        BLOCK3_FC2_TARGETS,
        "目标 block3 fc2 MatMul",
    )
    resize_names = [target[0] for target in v5.GAM_RESIZE_TARGETS.values()]
    v5.require_nodes(source_model, resize_names, "GAM Resize")
    v5.require_nodes(
        source_model,
        v5.ENCODER_SHARED_CONSUMERS,
        "Encoder 共享分支",
    )

    print("\n[INFO] 开始 ModelOpt INT8 Q/DQ 转换")
    quantize(
        onnx_path=str(input_path),
        output_path=str(temporary_path),
        quantize_mode="int8",
        calibration_method="entropy",
        calibration_cache_path=str(cache_path),
        calibration_eps=["cpu"],
        op_types_to_quantize=["Conv", "MatMul"],
        nodes_to_quantize=v5.exact_patterns(ALL_WEIGHTED_TARGETS),
        nodes_to_exclude=v5.exact_patterns(v5.ENCODER_SHARED_CONSUMERS),
        high_precision_dtype=args.high_precision_dtype,
        use_external_data_format=False,
        keep_intermediate_files=False,
        log_level="INFO",
    )

    if not temporary_path.is_file():
        raise RuntimeError("ModelOpt 未生成中间模型: {}".format(temporary_path))

    quantized_model = onnx.load(str(temporary_path))
    onnx.checker.check_model(quantized_model)
    v5.require_nodes(
        quantized_model,
        v5.TARGET_CONVS,
        "量化后的目标 Decoder Conv",
    )
    v5.require_nodes(
        quantized_model,
        ALL_MATMUL_TARGETS,
        "量化后的目标 MatMul",
    )
    v5.require_nodes(
        quantized_model,
        resize_names,
        "量化后的 GAM Resize",
    )
    v5.require_nodes(
        quantized_model,
        v5.ENCODER_SHARED_CONSUMERS,
        "量化后的 Encoder 共享分支",
    )

    cache_scales = import_scales_from_calib_cache(str(cache_path))
    v5.isolate_target_conv_qdq(quantized_model)
    isolate_matmul_activation_qdq(quantized_model, ALL_MATMUL_TARGETS)

    for module_name, (resize_name, conv_name) in v5.GAM_RESIZE_TARGETS.items():
        v5.propagate_gam_qdq_to_resize(
            quantized_model,
            cache_scales,
            module_name,
            resize_name,
            conv_name,
        )

    validate_quantized_model(quantized_model)

    onnx.save_model(
        quantized_model,
        str(output_path),
        save_as_external_data=False,
    )

    # 保存后重新读取校验，避免序列化阶段破坏图结构。
    saved_model = onnx.load(str(output_path))
    validate_quantized_model(saved_model)

    if args.keep_temporary:
        print("\n[INFO] 已保留中间模型: {}".format(temporary_path))
    else:
        temporary_path.unlink()
        print("\n[INFO] 已删除中间模型: {}".format(temporary_path))

    print("\n[DONE] {}".format(output_path))


if __name__ == "__main__":
    main()
