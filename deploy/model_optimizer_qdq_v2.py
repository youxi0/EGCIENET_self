#!/usr/bin/env python3

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import onnx
from onnx import ModelProto, NodeProto
from modelopt.onnx.quantization import quantize


# 12 个 FEM Conv + 3 个 GAM conv_pre，共 15 个。
TARGET_CONVS = [
    *[
        "/model/decoder/FEM{}/conv{}/Conv".format(fem, conv)
        for fem in range(1, 5)
        for conv in range(1, 4)
    ],
    "/model/decoder/d31/conv_pre/conv_pre.0/Conv",
    "/model/decoder/d42/conv_pre/conv_pre.0/Conv",
    "/model/decoder/d42_31/conv_pre/conv_pre.0/Conv",
]

# 这些 Encoder Conv 与 FEM 输入共享上游激活。
ENCODER_SHARED_CONSUMERS = [
    "/model/patch_embed2/proj/Conv",
    "/model/patch_embed3/proj/Conv",
    "/model/patch_embed4/proj/Conv",
]


def exact_patterns(names: Iterable[str]) -> List[str]:
    """把节点名转换成严格的正则表达式，避免 '.' 等字符扩大匹配范围。"""
    return [
        "^{}$".format(re.escape(name))
        for name in names
    ]


def build_node_maps(model: ModelProto):
    nodes_by_name: Dict[str, NodeProto] = {}
    producer: Dict[str, NodeProto] = {}
    consumers = defaultdict(list)

    for node in model.graph.node:
        if node.name:
            if node.name in nodes_by_name:
                raise RuntimeError("发现重复节点名: {}".format(node.name))
            nodes_by_name[node.name] = node

        for output_name in node.output:
            if output_name:
                producer[output_name] = node

        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)

    return nodes_by_name, producer, consumers


def require_nodes(
    model: ModelProto,
    required_names: Iterable[str],
    description: str,
) -> None:
    nodes_by_name, _, _ = build_node_maps(model)

    missing = sorted(
        set(required_names) - set(nodes_by_name)
    )

    if missing:
        raise RuntimeError(
            "{}节点不存在:\n  {}".format(
                description,
                "\n  ".join(missing),
            )
        )

def isolate_target_conv_qdq(model: ModelProto) -> None:
    """
    保证每个目标 Conv 的激活 DQ 输出只供该 Conv 使用。

    修改前：
        raw -> Q -> DQ --+-> target Conv
                         +-> Add
                         +-> Add_1
                         +-> patch_embed

    修改后：
        raw -> Q -> DQ ----> target Conv
          +----------------> Add
          +----------------> Add_1
          +----------------> patch_embed
    """
    target_set = set(TARGET_CONVS)
    nodes_by_name, producer, consumers = build_node_maps(model)

    print("\n[INFO] 隔离 15 个目标 Conv 的激活 Q/DQ")

    for target_name in TARGET_CONVS:
        conv = nodes_by_name[target_name]

        if conv.op_type != "Conv":
            raise RuntimeError(
                "{} 不是 Conv，而是 {}".format(
                    target_name,
                    conv.op_type,
                )
            )

        if len(conv.input) < 2:
            raise RuntimeError(
                "{} 输入数量异常".format(target_name)
            )

        activation_dq = producer.get(conv.input[0])
        if (
            activation_dq is None
            or activation_dq.op_type != "DequantizeLinear"
        ):
            raise RuntimeError(
                "{} 的激活输入不是直接 DQ: {}".format(
                    target_name,
                    conv.input[0],
                )
            )

        q = producer.get(activation_dq.input[0])
        if q is None or q.op_type != "QuantizeLinear":
            raise RuntimeError(
                "{} 前面没有完整的 Q -> DQ".format(target_name)
            )

        raw_activation = q.input[0]
        dq_output = activation_dq.output[0]

        dq_consumers = list(consumers.get(dq_output, []))

        target_consumers = [
            node
            for node in dq_consumers
            if node.name in target_set
        ]

        # 当前模型预期每个激活 DQ 只对应一个目标 Conv。
        # 如果多个目标 Conv 共用同一个 DQ，需要克隆 Q/DQ，不能直接重连。
        if len(target_consumers) != 1:
            raise RuntimeError(
                "{} 的 DQ 被 {} 个目标 Conv 共用: {}".format(
                    target_name,
                    len(target_consumers),
                    [
                        node.name
                        for node in target_consumers
                    ],
                )
            )

        if target_consumers[0].name != target_name:
            raise RuntimeError(
                "{} 的 DQ 目标消费者不匹配: {}".format(
                    target_name,
                    target_consumers[0].name,
                )
            )

        rewired = []

        # 所有非目标消费者绕过 Q/DQ，接回原始 FP16 激活。
        for consumer in dq_consumers:
            if consumer is conv:
                continue

            for input_index, input_name in enumerate(consumer.input):
                if input_name != dq_output:
                    continue

                consumer.input[input_index] = raw_activation

                rewired.append(
                    "{}[input {}]".format(
                        consumer.name or "<unnamed>",
                        input_index,
                    )
                )

        if rewired:
            print(
                "[REWIRE] {}\n"
                "         DQ: {}\n"
                "         RAW: {}\n"
                "         bypass consumers:\n"
                "           {}".format(
                    target_name,
                    dq_output,
                    raw_activation,
                    "\n           ".join(rewired),
                )
            )
        else:
            print(
                "[KEEP] {} 的 DQ 已经是专用分支".format(
                    target_name
                )
            )


def is_weight_quantized(
    node: NodeProto,
    producer: Dict[str, NodeProto],
) -> bool:
    if node.op_type != "Conv" or len(node.input) < 2:
        return False

    weight_producer = producer.get(node.input[1])

    return (
        weight_producer is not None
        and weight_producer.op_type == "DequantizeLinear"
    )


def activation_comes_from_dq(
    node: NodeProto,
    producer: Dict[str, NodeProto],
) -> bool:
    if not node.input:
        return False

    activation_producer = producer.get(node.input[0])

    return (
        activation_producer is not None
        and activation_producer.op_type == "DequantizeLinear"
    )


def validate_quantized_model(model: ModelProto) -> None:
    onnx.checker.check_model(model)

    nodes_by_name, producer, _ = build_node_maps(model)
    expected = set(TARGET_CONVS)

    weight_quantized_convs: Set[str] = {
        node.name
        for node in model.graph.node
        if is_weight_quantized(node, producer)
    }

    missing = sorted(expected - weight_quantized_convs)
    unexpected = sorted(weight_quantized_convs - expected)

    encoder_dq_leaks = sorted(
        name
        for name in ENCODER_SHARED_CONSUMERS
        if activation_comes_from_dq(
            nodes_by_name[name],
            producer,
        )
    )

    qdq_count = Counter(
        node.op_type
        for node in model.graph.node
        if node.op_type in {
            "QuantizeLinear",
            "DequantizeLinear",
        }
    )

    print("\n[VALIDATION]")
    print("Q/DQ nodes: {}".format(dict(qdq_count)))
    print(
        "Weight-quantized Conv: {}".format(
            len(weight_quantized_convs)
        )
    )

    for name in sorted(weight_quantized_convs):
        print("  {}".format(name))

    print("Missing target Conv: {}".format(missing))
    print("Unexpected quantized Conv: {}".format(unexpected))
    print("Encoder activation DQ leaks: {}".format(encoder_dq_leaks))

    if missing:
        raise RuntimeError(
            "目标卷积未被量化: {}".format(missing)
        )

    if unexpected:
        raise RuntimeError(
            "出现非目标权重量化卷积: {}".format(unexpected)
        )

    if encoder_dq_leaks:
        raise RuntimeError(
            "Encoder 仍然使用共享 DQ 激活: {}".format(
                encoder_dq_leaks
            )
        )

    if weight_quantized_convs != expected:
        raise RuntimeError(
            "量化卷积集合与预期不一致"
        )

    print(
        "\n[PASS] 15 个 Decoder Conv 已量化，"
        "Encoder 共享分支已隔离"
    )
    shared_target_dq = []

    for target_name in TARGET_CONVS:
        conv = nodes_by_name[target_name]
        dq = producer.get(conv.input[0])

        if dq is None or dq.op_type != "DequantizeLinear":
            shared_target_dq.append(
                "{}: no activation DQ".format(target_name)
            )
            continue

        dq_consumers = [
            node.name or "<unnamed>"
            for node in model.graph.node
            if dq.output[0] in node.input
        ]

        if dq_consumers != [target_name]:
            shared_target_dq.append(
                "{}: {}".format(
                    target_name,
                    dq_consumers,
                )
            )

    print("Shared target DQ:", shared_target_dq)

    if shared_target_dq:
        raise RuntimeError(
            "目标 Conv 的激活 DQ 不是专用分支: {}".format(
                shared_target_dq
            )
        )



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "为 EGCIENet 的 15 个 Decoder Conv 生成显式 INT8 Q/DQ ONNX，"
            "并隔离 Encoder 的共享激活分支。"
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
        default=(
            "deploy/"
            "egcienet_352_multiclass_qdq_v3_branch_isolated.onnx"
        ),
        help="最终显式 Q/DQ ONNX",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已有输出文件",
    )

    parser.add_argument(
        "--keep-temporary",
        action="store_true",
        help="保留 ModelOpt 尚未隔离分支的中间模型",
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
        raise FileNotFoundError(
            "找不到输入模型: {}".format(input_path)
        )

    if not cache_path.is_file():
        raise FileNotFoundError(
            "找不到 calibration cache: {}".format(cache_path)
        )

    if input_path.resolve() == output_path.resolve():
        raise RuntimeError(
            "输出路径不能覆盖原始 ONNX"
        )

    if output_path.exists() and not args.force:
        raise FileExistsError(
            "输出已经存在: {}\n"
            "如需覆盖，请增加 --force".format(output_path)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if temporary_path.exists():
        temporary_path.unlink()

    print("[INPUT ] {}".format(input_path))
    print("[CACHE ] {}".format(cache_path))
    print("[TEMP  ] {}".format(temporary_path))
    print("[OUTPUT] {}".format(output_path))

    source_model = onnx.load(str(input_path))
    onnx.checker.check_model(source_model)

    require_nodes(
        source_model,
        TARGET_CONVS,
        "目标 Decoder Conv",
    )
    require_nodes(
        source_model,
        ENCODER_SHARED_CONSUMERS,
        "Encoder 共享分支",
    )

    print("\n[INFO] 精确目标卷积，共 {} 个".format(len(TARGET_CONVS)))
    for name in TARGET_CONVS:
        print("  {}".format(name))

    print("\n[INFO] 开始 ModelOpt INT8 Q/DQ 转换")

    quantize(
        onnx_path=str(input_path),
        output_path=str(temporary_path),
        quantize_mode="int8",
        calibration_method="entropy",
        calibration_cache_path=str(cache_path),

        # 当前服务器 ORT CUDA 依赖不匹配，直接使用 CPU，
        # 避免尝试 TensorRT/CUDA EP 后产生无关报错。
        calibration_eps=["cpu"],

        op_types_to_quantize=["Conv"],

        # 使用 ^...$ 严格匹配，避免节点名中的 '.' 被当作正则通配符。
        nodes_to_quantize=exact_patterns(TARGET_CONVS),

        # 防御性排除；真正的共享激活隔离在量化后完成。
        nodes_to_exclude=exact_patterns(
            ENCODER_SHARED_CONSUMERS
        ),

        high_precision_dtype="fp16",
        use_external_data_format=False,
        keep_intermediate_files=False,
        log_level="INFO",
    )

    if not temporary_path.is_file():
        raise RuntimeError(
            "ModelOpt 没有生成中间模型: {}".format(
                temporary_path
            )
        )

    print("\n[INFO] 加载 ModelOpt 输出")
    quantized_model = onnx.load(str(temporary_path))
    onnx.checker.check_model(quantized_model)

    require_nodes(
        quantized_model,
        TARGET_CONVS,
        "量化后的目标 Decoder Conv",
    )
    require_nodes(
        quantized_model,
        ENCODER_SHARED_CONSUMERS,
        "量化后的 Encoder 共享分支",
    )

    isolate_target_conv_qdq(quantized_model)
    validate_quantized_model(quantized_model)

    onnx.save_model(
        quantized_model,
        str(output_path),
        save_as_external_data=False,
    )

    # 保存后重新读取验证，防止序列化问题。
    saved_model = onnx.load(str(output_path))
    validate_quantized_model(saved_model)

    if not args.keep_temporary:
        temporary_path.unlink()
        print(
            "\n[INFO] 已删除中间模型: {}".format(
                temporary_path
            )
        )
    else:
        print(
            "\n[INFO] 已保留中间模型: {}".format(
                temporary_path
            )
        )

    print("\n[DONE] {}".format(output_path))


if __name__ == "__main__":
    main()