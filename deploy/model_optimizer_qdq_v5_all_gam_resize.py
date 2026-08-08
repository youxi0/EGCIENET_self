#!/usr/bin/env python3

import argparse
import re
from collections import Counter, defaultdict
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, numpy_helper
from modelopt.onnx.quantization import quantize
from modelopt.onnx.quantization.calib_utils import import_scales_from_calib_cache


# 与 v3 保持一致：12 个 FEM Conv + 3 个 GAM conv_pre，共 15 个。
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

ENCODER_SHARED_CONSUMERS = [
    "/model/patch_embed2/proj/Conv",
    "/model/patch_embed3/proj/Conv",
    "/model/patch_embed4/proj/Conv",
]

# v5 将三个 GAM 的激活 Q/DQ 都扩展到各自 Resize 输入。
GAM_RESIZE_TARGETS = {
    "d31": (
        "/model/decoder/d31/Resize",
        "/model/decoder/d31/conv_pre/conv_pre.0/Conv",
    ),
    "d42": (
        "/model/decoder/d42/Resize",
        "/model/decoder/d42/conv_pre/conv_pre.0/Conv",
    ),
    "d42_31": (
        "/model/decoder/d42_31/Resize",
        "/model/decoder/d42_31/conv_pre/conv_pre.0/Conv",
    ),
}


def exact_patterns(names: Iterable[str]) -> List[str]:
    """生成严格匹配节点名的正则，避免节点名中的 '.' 扩大匹配。"""
    return ["^{}$".format(re.escape(name)) for name in names]


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
    missing = sorted(set(required_names) - set(nodes_by_name))

    if missing:
        raise RuntimeError(
            "{}节点不存在:\n  {}".format(
                description,
                "\n  ".join(missing),
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


def isolate_target_conv_qdq(model: ModelProto) -> None:
    """
    让每个目标 Conv 的激活 DQ 只供该 Conv 使用。

    非目标消费者（Add、Add_1、patch_embed 等）接回 Q 之前的
    高精度张量，避免共享 DQ 污染其他分支。
    """
    target_set = set(TARGET_CONVS)
    nodes_by_name, producer, consumers = build_node_maps(model)

    print("\n[INFO] 隔离 15 个目标 Conv 的激活 Q/DQ")

    for target_name in TARGET_CONVS:
        conv = nodes_by_name[target_name]

        if conv.op_type != "Conv" or len(conv.input) < 2:
            raise RuntimeError("{} 不是合法 Conv".format(target_name))

        activation_dq = producer.get(conv.input[0])
        if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
            raise RuntimeError(
                "{} 的激活输入不是直接 DQ: {}".format(
                    target_name,
                    conv.input[0],
                )
            )

        q = producer.get(activation_dq.input[0])
        if q is None or q.op_type != "QuantizeLinear":
            raise RuntimeError("{} 前面没有完整 Q -> DQ".format(target_name))

        raw_activation = q.input[0]
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
            if consumer.name == conv.name:
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
            print(
                "[REWIRE] {} bypass: {}".format(
                    target_name,
                    ", ".join(rewired),
                )
            )
        else:
            print("[KEEP] {} 的激活 DQ 已专用".format(target_name))


def find_cache_scale(
    module_name: str,
    tensor_name: str,
    producer: Dict[str, NodeProto],
    cache_scales: Dict[str, float],
) -> Tuple[str, float]:
    """
    查找 Resize 输入激活的校准 scale。

    ModelOpt 的 calibration cache 导入函数使用
    '<tensor_name>_scale' 作为键。若 FP16 autocast 在输入前插入 Cast，
    同时尝试 Cast 之前的原始张量名；不做静默回退。
    """
    candidates = [tensor_name + "_scale"]
    tensor_producer = producer.get(tensor_name)

    if (
        tensor_producer is not None
        and tensor_producer.op_type in {"Cast", "Identity"}
    ):
        if tensor_producer.input:
            candidates.append(tensor_producer.input[0] + "_scale")

    for key in candidates:
        if key in cache_scales:
            scale = float(cache_scales[key])
            if not np.isfinite(scale) or scale <= 0.0:
                raise RuntimeError(
                    "calibration cache 中的 scale 非法: {}={}".format(
                        key,
                        scale,
                    )
                )
            return key, scale

    suffix = tensor_name.rsplit("/", 1)[-1]
    nearby = sorted(
        key
        for key in cache_scales
        if suffix in key or module_name in key
    )[:30]

    raise RuntimeError(
        "找不到 {} Resize 输入张量的 calibration scale。\n"
        "tensor: {}\n"
        "tried: {}\n"
        "nearby cache keys:\n  {}".format(
            module_name,
            tensor_name,
            candidates,
            "\n  ".join(nearby) if nearby else "<none>",
        )
    )


def clone_scalar_initializer(
    template: onnx.TensorProto,
    name: str,
    value: float,
) -> onnx.TensorProto:
    """按现有 Q/DQ scale 的 dtype/shape 创建新标量 initializer。"""
    template_array = numpy_helper.to_array(template)

    if template_array.size != 1:
        raise RuntimeError(
            "激活 scale 预期为 per-tensor 标量，实际 shape={}".format(
                template_array.shape
            )
        )

    array = np.full(
        template_array.shape,
        value,
        dtype=template_array.dtype,
    )
    return numpy_helper.from_array(array, name=name)


def propagate_gam_qdq_to_resize(
    model: ModelProto,
    cache_scales: Dict[str, float],
    module_name: str,
    resize_name: str,
    conv_name: str,
) -> None:
    """
    在指定 GAM Resize 输入前插入 Q/DQ，保留 Resize 输出后的原 Q/DQ。

    修改前：
        raw -> Resize -> Q_out -> DQ_out -> GAM conv_pre

    修改后：
        raw -> Q_in -> DQ_in -> Resize -> Q_out -> DQ_out -> GAM conv_pre

    TensorRT 因而可以把位于 DQ_in 与 Q_out 之间的 Resize 作为
    INT8 层处理。GAM Add 仍在 conv_pre 输出后的高精度路径。
    """
    nodes_by_name, producer, consumers = build_node_maps(model)
    initializers = {init.name: init for init in model.graph.initializer}

    resize = nodes_by_name[resize_name]
    conv = nodes_by_name[conv_name]

    if resize.op_type != "Resize":
        raise RuntimeError("{} 不是 Resize".format(resize_name))
    if not resize.input or not resize.output:
        raise RuntimeError("{} Resize 输入或输出为空".format(module_name))

    conv_input_dq = producer.get(conv.input[0])
    if conv_input_dq is None or conv_input_dq.op_type != "DequantizeLinear":
        raise RuntimeError("{} conv_pre 输入不是 DQ".format(module_name))

    resize_output_q = producer.get(conv_input_dq.input[0])
    if resize_output_q is None or resize_output_q.op_type != "QuantizeLinear":
        raise RuntimeError("{} conv_pre 前没有 Resize 输出 Q/DQ".format(module_name))

    if resize_output_q.input[0] != resize.output[0]:
        raise RuntimeError(
            "{} Q/DQ 不直接位于 Resize 输出：Q input={}, Resize output={}".format(
                module_name,
                resize_output_q.input[0],
                resize.output[0],
            )
        )

    resize_output_consumers = consumers.get(resize.output[0], [])
    if [node.name for node in resize_output_consumers] != [resize_output_q.name]:
        raise RuntimeError(
            "{} Resize 输出不是 Q 专用输入: {}".format(
                module_name,
                [node.name for node in resize_output_consumers]
            )
        )

    output_q_consumers = consumers.get(resize_output_q.output[0], [])
    if [node.name for node in output_q_consumers] != [conv_input_dq.name]:
        raise RuntimeError(
            "{} Resize 输出 Q 被异常共享: {}".format(
                module_name,
                [node.name for node in output_q_consumers]
            )
        )

    raw_resize_input = resize.input[0]
    existing_input_dq = producer.get(raw_resize_input)
    if existing_input_dq is not None and existing_input_dq.op_type == "DequantizeLinear":
        existing_input_q = producer.get(existing_input_dq.input[0])
        if existing_input_q is not None and existing_input_q.op_type == "QuantizeLinear":
            print("[KEEP] {} Resize 输入前已经存在 Q/DQ".format(module_name))
            return

    cache_key, input_scale_value = find_cache_scale(
        module_name,
        raw_resize_input,
        producer,
        cache_scales,
    )

    if len(resize_output_q.input) < 2 or len(conv_input_dq.input) < 2:
        raise RuntimeError("现有 {} 输出 Q/DQ 缺少 scale 输入".format(module_name))

    scale_template = initializers.get(resize_output_q.input[1])
    if scale_template is None:
        raise RuntimeError(
            "找不到现有 {} Q scale initializer: {}".format(
                module_name,
                resize_output_q.input[1]
            )
        )

    prefix = "/model/decoder/{}/Resize_input_v5".format(module_name)
    scale_name = prefix + "_scale"
    q_output_name = prefix + "_QuantizeLinear_Output"
    dq_output_name = prefix + "_DequantizeLinear_Output"

    occupied_tensor_names = set(producer) | set(initializers)
    occupied_node_names = set(nodes_by_name)
    for name in (scale_name, q_output_name, dq_output_name):
        if name in occupied_tensor_names:
            raise RuntimeError("准备创建的张量名已存在: {}".format(name))
    for name in (prefix + "_QuantizeLinear", prefix + "_DequantizeLinear"):
        if name in occupied_node_names:
            raise RuntimeError("准备创建的节点名已存在: {}".format(name))

    new_scale = clone_scalar_initializer(
        scale_template,
        scale_name,
        input_scale_value,
    )
    model.graph.initializer.append(new_scale)

    new_q = deepcopy(resize_output_q)
    new_q.name = prefix + "_QuantizeLinear"
    del new_q.input[:]
    new_q.input.extend([raw_resize_input, scale_name])
    del new_q.output[:]
    new_q.output.extend([q_output_name])

    new_dq = deepcopy(conv_input_dq)
    new_dq.name = prefix + "_DequantizeLinear"
    del new_dq.input[:]
    new_dq.input.extend([q_output_name, scale_name])
    del new_dq.output[:]
    new_dq.output.extend([dq_output_name])

    # 若现有激活 Q/DQ 显式使用 zero-point，则复制其 dtype/shape。
    if len(resize_output_q.input) >= 3:
        zp_template = initializers.get(resize_output_q.input[2])
        if zp_template is None:
            raise RuntimeError(
                "找不到现有 {} Q zero-point initializer: {}".format(
                    module_name,
                    resize_output_q.input[2]
                )
            )

        zero_point_name = prefix + "_zero_point"
        if zero_point_name in occupied_tensor_names:
            raise RuntimeError(
                "准备创建的 zero-point 名已存在: {}".format(zero_point_name)
            )
        zero_point = deepcopy(zp_template)
        zero_point.name = zero_point_name
        model.graph.initializer.append(zero_point)
        new_q.input.extend([zero_point_name])
        new_dq.input.extend([zero_point_name])

    resize.input[0] = dq_output_name

    # Q/DQ 必须排在 Resize 前，保持 ONNX 拓扑顺序。
    reordered_nodes = []
    inserted = False
    for node in model.graph.node:
        if node.name == resize_name:
            reordered_nodes.extend([new_q, new_dq])
            inserted = True
        reordered_nodes.append(node)

    if not inserted:
        raise RuntimeError("无法在图中定位 {} Resize".format(module_name))

    del model.graph.node[:]
    model.graph.node.extend(reordered_nodes)

    print("\n[PROPAGATE] {} Q/DQ 已扩展到 Resize".format(module_name))
    print("  raw input : {}".format(raw_resize_input))
    print("  cache key : {}".format(cache_key))
    print("  input scale: {:.9g}".format(input_scale_value))
    print("  new chain : Q -> DQ -> {} -> Q -> DQ -> {}".format(
        resize_name,
        conv_name,
    ))


def validate_target_conv_qdq(model: ModelProto) -> None:
    nodes_by_name, producer, consumers = build_node_maps(model)
    expected = set(TARGET_CONVS)

    weight_quantized_convs: Set[str] = {
        node.name
        for node in model.graph.node
        if is_weight_quantized(node, producer)
    }

    missing = sorted(expected - weight_quantized_convs)
    unexpected = sorted(weight_quantized_convs - expected)
    shared_target_dq = []

    for target_name in TARGET_CONVS:
        conv = nodes_by_name[target_name]
        dq = producer.get(conv.input[0])

        if dq is None or dq.op_type != "DequantizeLinear":
            shared_target_dq.append("{}: no activation DQ".format(target_name))
            continue

        dq_consumers = consumers.get(dq.output[0], [])
        if (
            len(dq_consumers) != 1
            or dq_consumers[0].name != conv.name
        ):
            shared_target_dq.append(
                "{}: {}".format(
                    target_name,
                    [node.name for node in dq_consumers],
                )
            )

    encoder_dq_leaks = []
    for name in ENCODER_SHARED_CONSUMERS:
        node = nodes_by_name[name]
        parent = producer.get(node.input[0])
        if parent is not None and parent.op_type == "DequantizeLinear":
            encoder_dq_leaks.append(name)

    if missing:
        raise RuntimeError("目标卷积未量化: {}".format(missing))
    if unexpected:
        raise RuntimeError("出现非目标权重量化卷积: {}".format(unexpected))
    if shared_target_dq:
        raise RuntimeError("目标 Conv 的激活 DQ 非专用: {}".format(shared_target_dq))
    if encoder_dq_leaks:
        raise RuntimeError("Encoder 仍使用共享 DQ: {}".format(encoder_dq_leaks))


def validate_gam_resize_island(
    model: ModelProto,
    module_name: str,
    resize_name: str,
    conv_name: str,
) -> None:
    nodes_by_name, producer, consumers = build_node_maps(model)
    initializers = {init.name: init for init in model.graph.initializer}
    resize = nodes_by_name[resize_name]
    conv = nodes_by_name[conv_name]

    input_dq = producer.get(resize.input[0])
    if input_dq is None or input_dq.op_type != "DequantizeLinear":
        raise RuntimeError("{} Resize 输入前没有 DQ".format(module_name))

    input_q = producer.get(input_dq.input[0])
    if input_q is None or input_q.op_type != "QuantizeLinear":
        raise RuntimeError(
            "{} Resize 输入前没有完整 Q -> DQ".format(module_name)
        )

    input_dq_consumers = consumers.get(input_dq.output[0], [])
    if [node.name for node in input_dq_consumers] != [resize.name]:
        raise RuntimeError(
            "{} Resize 输入 DQ 不是专用分支: {}".format(
                module_name,
                [node.name for node in input_dq_consumers]
            )
        )

    resize_output_consumers = consumers.get(resize.output[0], [])
    if (
        len(resize_output_consumers) != 1
        or resize_output_consumers[0].op_type != "QuantizeLinear"
    ):
        raise RuntimeError(
            "{} Resize 输出没有唯一专用 Q: {}".format(
                module_name,
                [node.name for node in resize_output_consumers]
            )
        )

    output_q = resize_output_consumers[0]
    output_q_consumers = consumers.get(output_q.output[0], [])
    if (
        len(output_q_consumers) != 1
        or output_q_consumers[0].op_type != "DequantizeLinear"
    ):
        raise RuntimeError(
            "{} Resize 输出 Q 后没有唯一专用 DQ: {}".format(
                module_name,
                [node.name for node in output_q_consumers]
            )
        )

    output_dq = output_q_consumers[0]
    output_dq_consumers = consumers.get(output_dq.output[0], [])
    if [node.name for node in output_dq_consumers] != [conv.name]:
        raise RuntimeError(
            "{} Resize 输出 DQ 未直接专供 conv_pre: {}".format(
                module_name,
                [node.name for node in output_dq_consumers]
            )
        )

    input_scale = initializers.get(input_q.input[1])
    if input_scale is None:
        raise RuntimeError(
            "{} Resize 输入 Q 的 scale 不存在".format(module_name)
        )

    input_scale_value = float(np.asarray(numpy_helper.to_array(input_scale)).reshape(-1)[0])
    if not np.isfinite(input_scale_value) or input_scale_value <= 0.0:
        raise RuntimeError(
            "{} Resize 输入 Q 的 scale 非法".format(module_name)
        )

    print("\n[VALIDATION] {}".format(module_name))
    print("  input chain : Q -> DQ -> Resize")
    print("  output chain: Resize -> Q -> DQ -> conv_pre")
    print("  input scale : {:.9g}".format(input_scale_value))


def validate_quantized_model(model: ModelProto) -> None:
    onnx.checker.check_model(model)
    validate_target_conv_qdq(model)
    for module_name, (resize_name, conv_name) in GAM_RESIZE_TARGETS.items():
        validate_gam_resize_island(
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
    print("  Q/DQ nodes      : {}".format(dict(qdq_count)))
    print("  INT8 target Conv: 15/15")
    print("\n[PASS] d31、d42、d42_31 Resize 均已进入显式 INT8 Q/DQ 岛")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "基于 v3 的 15-Conv Q/DQ 模型，将 d31、d42、d42_31 "
            "的 Q/DQ 向上游扩展到各自的 bilinear Resize。"
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
        default="deploy/egcienet_352_multiclass_qdq_v5_all_gam_resize.onnx",
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
            "输出已经存在: {}，如需覆盖请增加 --force".format(output_path)
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

    source_model = onnx.load(str(input_path))
    onnx.checker.check_model(source_model)
    require_nodes(source_model, TARGET_CONVS, "目标 Decoder Conv")
    resize_names = [target[0] for target in GAM_RESIZE_TARGETS.values()]
    require_nodes(source_model, resize_names, "GAM Resize")
    require_nodes(source_model, ENCODER_SHARED_CONSUMERS, "Encoder 共享分支")

    print("\n[INFO] 开始 ModelOpt INT8 Q/DQ 转换")
    quantize(
        onnx_path=str(input_path),
        output_path=str(temporary_path),
        quantize_mode="int8",
        calibration_method="entropy",
        calibration_cache_path=str(cache_path),
        calibration_eps=["cpu"],
        op_types_to_quantize=["Conv"],
        nodes_to_quantize=exact_patterns(TARGET_CONVS),
        nodes_to_exclude=exact_patterns(ENCODER_SHARED_CONSUMERS),
        high_precision_dtype=args.high_precision_dtype,
        use_external_data_format=False,
        keep_intermediate_files=False,
        log_level="INFO",
    )

    if not temporary_path.is_file():
        raise RuntimeError("ModelOpt 未生成中间模型: {}".format(temporary_path))

    quantized_model = onnx.load(str(temporary_path))
    onnx.checker.check_model(quantized_model)
    require_nodes(quantized_model, TARGET_CONVS, "量化后的目标 Decoder Conv")
    require_nodes(quantized_model, resize_names, "量化后的 GAM Resize")
    require_nodes(
        quantized_model,
        ENCODER_SHARED_CONSUMERS,
        "量化后的 Encoder 共享分支",
    )

    cache_scales = import_scales_from_calib_cache(str(cache_path))
    isolate_target_conv_qdq(quantized_model)
    for module_name, (resize_name, conv_name) in GAM_RESIZE_TARGETS.items():
        propagate_gam_qdq_to_resize(
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

    # 重新读取并验证，防止保存阶段破坏图结构。
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
