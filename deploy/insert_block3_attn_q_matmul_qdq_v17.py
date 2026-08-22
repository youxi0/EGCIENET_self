#!/usr/bin/env python3

"""
V17（修正版）：在已经完成 Plugin 改图的 V16 ONNX 上，只量化
block3.0～block3.17 的 attention q/MatMul。

本脚本不会再次调用 ModelOpt 改写整张图。它只使用 ModelOpt 的校准
cache 读取函数取得激活 scale，然后直接向 V16 图中插入显式 Q/DQ。
这样可以避免 ModelOpt 重新处理 V16 中的 block1/block3/block4 自定义
Plugin 节点。

每个 q/MatMul 都插入两条与现有 block3 fc1 完全同构的 Q/DQ：

  norm1 输出 -> 激活 Q -> 激活 DQ --+
                                      +-> q/MatMul -> FP16 attention 核心
  q 静态权重 -> 权重 Q -> 权重 DQ --+

边界约束：
  - 激活 DQ 只允许对应的 q/MatMul 消费；
  - 权重采用 axis=1 的逐输出通道对称 INT8 量化；
  - SR、KV、QK^T、Softmax、Attention×V、projection 和残差保持不变；
  - V16 中已有的 Q/DQ 和 egcinet Plugin 节点必须原样保留。
"""

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper


BLOCK3_Q_TARGETS: List[str] = [
    "/model/block3.{}/attn/q/MatMul".format(index)
    for index in range(18)
]

BLOCK3_FC1_TEMPLATES: List[str] = [
    "/model/block3.{}/mlp/fc1/MatMul".format(index)
    for index in range(18)
]


def build_maps(
    model: ModelProto,
) -> Tuple[
    Dict[str, NodeProto],
    Dict[str, NodeProto],
    Dict[str, List[NodeProto]],
    Dict[str, TensorProto],
]:
    """建立节点、生产者、消费者和 initializer 索引。"""
    nodes_by_name: Dict[str, NodeProto] = {}
    producer: Dict[str, NodeProto] = {}
    consumers: Dict[str, List[NodeProto]] = {}

    for node in model.graph.node:
        if not node.name:
            raise RuntimeError("图中存在没有名称的节点，无法安全做定点改图")
        if node.name in nodes_by_name:
            raise RuntimeError("节点名称重复: {}".format(node.name))
        nodes_by_name[node.name] = node

        for output_name in node.output:
            if not output_name:
                continue
            if output_name in producer:
                raise RuntimeError("tensor 存在多个生产者: {}".format(output_name))
            producer[output_name] = node

        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)

    initializers = {
        initializer.name: initializer
        for initializer in model.graph.initializer
    }
    return nodes_by_name, producer, consumers, initializers


def require_node(
    nodes_by_name: Mapping[str, NodeProto],
    name: str,
    op_type: str,
) -> NodeProto:
    """按名称和算子类型严格取得一个节点。"""
    node = nodes_by_name.get(name)
    if node is None:
        raise RuntimeError("找不到目标节点: {}".format(name))
    if node.op_type != op_type:
        raise RuntimeError(
            "{} 类型错误，期望 {}，实际 {}".format(
                name,
                op_type,
                node.op_type,
            )
        )
    return node


def get_axis(node: NodeProto) -> int:
    """读取 Q/DQ 的 axis 属性；本脚本只接受明确的逐通道 axis。"""
    for attribute in node.attribute:
        if attribute.name == "axis":
            return int(helper.get_attribute_value(attribute))
    raise RuntimeError("{} 缺少 axis 属性".format(node.name))


def initializer_array(
    initializers: Mapping[str, TensorProto],
    name: str,
) -> np.ndarray:
    """读取 initializer，并在缺失时给出明确错误。"""
    initializer = initializers.get(name)
    if initializer is None:
        raise RuntimeError("找不到 initializer: {}".format(name))
    return np.asarray(numpy_helper.to_array(initializer))


def plugin_snapshot(model: ModelProto) -> Dict[str, str]:
    """记录所有 egcinet Plugin 节点的序列化摘要，防止改图污染 Plugin。"""
    snapshot: Dict[str, str] = {}
    for node in model.graph.node:
        if node.domain != "egcinet":
            continue
        snapshot[node.name] = hashlib.sha256(
            node.SerializeToString()
        ).hexdigest()
    return snapshot


def qdq_count(model: ModelProto) -> Counter:
    """统计显式 Q/DQ 节点数量。"""
    return Counter(
        node.op_type
        for node in model.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
    )


def load_cache_scales(cache_path: Path) -> Dict[str, float]:
    """
    使用当前 ModelOpt 解析 TensorRT calibration cache。

    这里只读取 scale，不让 ModelOpt 接触或改写带 Plugin 的 V16 ONNX。
    """
    try:
        from modelopt.onnx.quantization.calib_utils import (
            import_scales_from_calib_cache,
        )
    except ImportError as error:
        raise ImportError(
            "无法导入 ModelOpt calibration cache 读取函数；请在已经安装 "
            "nvidia-modelopt[onnx] 的环境中运行本脚本"
        ) from error

    raw_scales = import_scales_from_calib_cache(str(cache_path))
    scales = {
        str(key): float(value)
        for key, value in raw_scales.items()
    }
    if not scales:
        raise RuntimeError("calibration cache 中没有任何 scale")
    return scales


def find_activation_scale(
    tensor_name: str,
    producer: Mapping[str, NodeProto],
    cache_scales: Mapping[str, float],
) -> Tuple[str, float]:
    """
    查找 q/MatMul 原始激活的校准 scale。

    ModelOpt 的 cache 导入函数通常使用 '<tensor_name>_scale' 作为键。
    同时兼容 tensor 前面或后面存在单层 Cast 的情况，但不会用模糊匹配
    自动选择 scale，避免把别的 block 的统计值误用到当前 block。
    """
    candidates: List[str] = [tensor_name + "_scale", tensor_name]

    tensor_producer = producer.get(tensor_name)
    if (
        tensor_producer is not None
        and tensor_producer.op_type == "Cast"
        and tensor_producer.input
    ):
        raw_name = tensor_producer.input[0]
        candidates.extend([raw_name + "_scale", raw_name])

    for key in candidates:
        if key not in cache_scales:
            continue
        scale = float(cache_scales[key])
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(
                "calibration scale 非法: {}={}".format(key, scale)
            )
        return key, scale

    # 失败时只打印相关键，便于检查 cache 的真实命名，不做危险的自动猜测。
    tail = tensor_name.split("/")[-2:]
    keyword = "/".join(tail)
    nearby = sorted(
        key
        for key in cache_scales
        if keyword in key or tensor_name in key
    )[:20]
    raise RuntimeError(
        "找不到激活 calibration scale:\n"
        "  tensor    : {}\n"
        "  tried     : {}\n"
        "  nearby    : {}".format(tensor_name, candidates, nearby)
    )


def make_initializer_like(
    template: TensorProto,
    name: str,
    value: np.ndarray,
) -> TensorProto:
    """沿用现有 block3 fc1 Q/DQ initializer 的数据类型。"""
    template_dtype = numpy_helper.to_array(template).dtype
    array = np.asarray(value, dtype=template_dtype)
    return numpy_helper.from_array(array, name=name)


def require_available_names(
    model: ModelProto,
    node_names: Iterable[str],
    tensor_names: Iterable[str],
) -> None:
    """在插入前检查新节点名和 tensor 名不会与 V16 冲突。"""
    existing_node_names = {node.name for node in model.graph.node}
    existing_tensor_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    existing_tensor_names.update(
        initializer.name for initializer in model.graph.initializer
    )

    node_collisions = sorted(set(node_names) & existing_node_names)
    tensor_collisions = sorted(set(tensor_names) & existing_tensor_names)
    if node_collisions or tensor_collisions:
        raise RuntimeError(
            "V17 Q/DQ 名称冲突:\n  nodes={}\n  tensors={}".format(
                node_collisions,
                tensor_collisions,
            )
        )


def inspect_fc1_template(
    block_index: int,
    nodes_by_name: Mapping[str, NodeProto],
    producer: Mapping[str, NodeProto],
    initializers: Mapping[str, TensorProto],
) -> Dict[str, object]:
    """
    读取同一 block 已经验证成功的 fc1 Q/DQ，作为 q/MatMul 的格式模板。

    这里不复用 fc1 的数值 scale，只复用节点结构、axis、scale dtype 和
    zero-point dtype。
    """
    fc1_name = BLOCK3_FC1_TEMPLATES[block_index]
    fc1 = require_node(nodes_by_name, fc1_name, "MatMul")
    if len(fc1.input) != 2:
        raise RuntimeError("{} 输入数量不是 2".format(fc1_name))

    activation_dq = producer.get(fc1.input[0])
    weight_dq = producer.get(fc1.input[1])
    if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
        raise RuntimeError("{} 缺少已有激活 DQ 模板".format(fc1_name))
    if weight_dq is None or weight_dq.op_type != "DequantizeLinear":
        raise RuntimeError("{} 缺少已有权重 DQ 模板".format(fc1_name))

    activation_q = producer.get(activation_dq.input[0])
    weight_q = producer.get(weight_dq.input[0])
    if activation_q is None or activation_q.op_type != "QuantizeLinear":
        raise RuntimeError("{} 缺少已有激活 Q 模板".format(fc1_name))
    if weight_q is None or weight_q.op_type != "QuantizeLinear":
        raise RuntimeError("{} 缺少已有权重 Q 模板".format(fc1_name))

    if len(activation_q.input) != 3 or len(activation_dq.input) != 3:
        raise RuntimeError("{} 激活 Q/DQ 不是 scale + zero-point 结构".format(fc1_name))
    if len(weight_q.input) != 3 or len(weight_dq.input) != 3:
        raise RuntimeError("{} 权重 Q/DQ 不是 scale + zero-point 结构".format(fc1_name))

    weight_axis_q = get_axis(weight_q)
    weight_axis_dq = get_axis(weight_dq)
    if weight_axis_q != 1 or weight_axis_dq != 1:
        raise RuntimeError(
            "{} 权重模板不是 axis=1: Q={}, DQ={}".format(
                fc1_name,
                weight_axis_q,
                weight_axis_dq,
            )
        )

    activation_scale_template = initializers.get(activation_q.input[1])
    activation_zero_template = initializers.get(activation_q.input[2])
    weight_scale_template = initializers.get(weight_q.input[1])
    weight_zero_template = initializers.get(weight_q.input[2])
    if any(
        item is None
        for item in (
            activation_scale_template,
            activation_zero_template,
            weight_scale_template,
            weight_zero_template,
        )
    ):
        raise RuntimeError("{} 的 Q/DQ 模板 initializer 不完整".format(fc1_name))

    activation_scale_array = numpy_helper.to_array(activation_scale_template)
    activation_zero_array = numpy_helper.to_array(activation_zero_template)
    weight_scale_array = numpy_helper.to_array(weight_scale_template)
    weight_zero_array = numpy_helper.to_array(weight_zero_template)

    if activation_scale_array.shape != () or activation_zero_array.shape != ():
        raise RuntimeError("{} 激活模板不是 per-tensor 标量".format(fc1_name))
    if activation_zero_array.dtype != np.int8:
        raise RuntimeError("{} 激活模板 zero-point 不是 INT8".format(fc1_name))
    if weight_scale_array.ndim != 1 or weight_zero_array.ndim != 1:
        raise RuntimeError("{} 权重模板不是一维逐通道参数".format(fc1_name))
    if weight_zero_array.dtype != np.int8:
        raise RuntimeError("{} 权重模板 zero-point 不是 INT8".format(fc1_name))

    return {
        "activation_scale": activation_scale_template,
        "activation_zero": activation_zero_template,
        "weight_scale": weight_scale_template,
        "weight_zero": weight_zero_template,
        "weight_axis": weight_axis_q,
    }


def create_qdq_for_target(
    model: ModelProto,
    block_index: int,
    target: NodeProto,
    producer: Mapping[str, NodeProto],
    consumers: Mapping[str, Sequence[NodeProto]],
    initializers: Mapping[str, TensorProto],
    cache_scales: Mapping[str, float],
    template: Mapping[str, object],
) -> Tuple[List[NodeProto], List[TensorProto], str, float]:
    """为一个 q/MatMul 创建专用激活 Q/DQ 和逐通道权重 Q/DQ。"""
    if len(target.input) != 2:
        raise RuntimeError("{} 输入数量不是 2".format(target.name))

    raw_activation = target.input[0]
    raw_weight = target.input[1]

    # V16 的 q/MatMul 必须还是未量化状态，防止重复插入 Q/DQ。
    activation_parent = producer.get(raw_activation)
    weight_parent = producer.get(raw_weight)
    if activation_parent is not None and activation_parent.op_type == "DequantizeLinear":
        raise RuntimeError("{} 激活已经有 DQ，拒绝重复插入".format(target.name))
    if weight_parent is not None and weight_parent.op_type == "DequantizeLinear":
        raise RuntimeError("{} 权重已经有 DQ，拒绝重复插入".format(target.name))

    weight_initializer = initializers.get(raw_weight)
    if weight_initializer is None:
        raise RuntimeError(
            "{} 的权重不是直接 initializer: {}".format(
                target.name,
                raw_weight,
            )
        )
    weight_consumers = list(consumers.get(raw_weight, []))
    if len(weight_consumers) != 1 or weight_consumers[0].name != target.name:
        raise RuntimeError(
            "{} 的权重不是专用权重: {}".format(
                target.name,
                [node.name for node in weight_consumers],
            )
        )

    weight = np.asarray(numpy_helper.to_array(weight_initializer))
    if weight.ndim != 2:
        raise RuntimeError(
            "{} 权重维度不是二维: {}".format(target.name, weight.shape)
        )
    output_channels = int(weight.shape[1])

    # 步骤 1：从校准 cache 取得当前 norm1 输出自己的激活 scale。
    cache_key, activation_scale = find_activation_scale(
        raw_activation,
        producer,
        cache_scales,
    )

    # 步骤 2：按 MatMul 权重 [K,N] 的 N 轴计算逐输出通道对称 scale。
    # 使用 127 而不是 128，保证正负两侧都落在 TensorRT 对称 INT8 范围内。
    weight_fp32 = weight.astype(np.float32, copy=False)
    weight_amax = np.max(np.abs(weight_fp32), axis=0)
    weight_scale = weight_amax / np.float32(127.0)
    weight_scale = np.maximum(weight_scale, np.float32(1.0e-8))
    if weight_scale.shape != (output_channels,):
        raise RuntimeError("{} 权重 scale shape 计算错误".format(target.name))
    if not np.all(np.isfinite(weight_scale)):
        raise RuntimeError("{} 权重 scale 存在非有限值".format(target.name))

    activation_prefix = target.name + "_V17_Activation"
    weight_prefix = target.name + "_V17_Weight"

    activation_scale_q_name = activation_prefix + "_scale_1"
    activation_scale_dq_name = activation_prefix + "_scale_2"
    activation_zero_q_name = activation_prefix + "_zero_point_1"
    activation_zero_dq_name = activation_prefix + "_zero_point_2"
    activation_q_output = activation_prefix + "_QuantizeLinear_Output"
    activation_dq_output = activation_prefix + "_DequantizeLinear_Output"

    weight_scale_q_name = weight_prefix + "_scale_1"
    weight_scale_dq_name = weight_prefix + "_scale_2"
    weight_zero_q_name = weight_prefix + "_zero_point_1"
    weight_zero_dq_name = weight_prefix + "_zero_point_2"
    weight_q_output = weight_prefix + "_QuantizeLinear_Output"
    weight_dq_output = weight_prefix + "_DequantizeLinear_Output"

    activation_q_name = activation_prefix + "_QuantizeLinear"
    activation_dq_name = activation_prefix + "_DequantizeLinear"
    weight_q_name = weight_prefix + "_QuantizeLinear"
    weight_dq_name = weight_prefix + "_DequantizeLinear"

    require_available_names(
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
    activation_zero_value = np.asarray(0, dtype=np.int8).reshape(())
    weight_zero_value = np.zeros((output_channels,), dtype=np.int8)

    new_initializers = [
        make_initializer_like(
            template["activation_scale"],
            activation_scale_q_name,
            activation_scale_value,
        ),
        make_initializer_like(
            template["activation_scale"],
            activation_scale_dq_name,
            activation_scale_value,
        ),
        make_initializer_like(
            template["activation_zero"],
            activation_zero_q_name,
            activation_zero_value,
        ),
        make_initializer_like(
            template["activation_zero"],
            activation_zero_dq_name,
            activation_zero_value,
        ),
        make_initializer_like(
            template["weight_scale"],
            weight_scale_q_name,
            weight_scale,
        ),
        make_initializer_like(
            template["weight_scale"],
            weight_scale_dq_name,
            weight_scale,
        ),
        make_initializer_like(
            template["weight_zero"],
            weight_zero_q_name,
            weight_zero_value,
        ),
        make_initializer_like(
            template["weight_zero"],
            weight_zero_dq_name,
            weight_zero_value,
        ),
    ]

    # 步骤 3：创建与现有 fc1 同构的四个 Q/DQ 节点。
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
        axis=int(template["weight_axis"]),
    )
    weight_dq = helper.make_node(
        "DequantizeLinear",
        inputs=[weight_q_output, weight_scale_dq_name, weight_zero_dq_name],
        outputs=[weight_dq_output],
        name=weight_dq_name,
        axis=int(template["weight_axis"]),
    )

    # 步骤 4：只替换 q/MatMul 的两个输入，不改它的输出和下游 Attention。
    target.input[0] = activation_dq_output
    target.input[1] = weight_dq_output

    print(
        "[INSERT] block3.{} q/MatMul: activation_scale={:.9g} "
        "cache_key={} weight_shape={} weight_axis=1".format(
            block_index,
            activation_scale,
            cache_key,
            tuple(weight.shape),
        )
    )
    return (
        [activation_q, activation_dq, weight_q, weight_dq],
        new_initializers,
        raw_activation,
        activation_scale,
    )


def is_quantized_weight_matmul(
    node: NodeProto,
    producer: Mapping[str, NodeProto],
) -> bool:
    """判断 MatMul 权重输入是否来自完整的 Q -> DQ。"""
    if node.op_type != "MatMul" or len(node.input) != 2:
        return False
    weight_dq = producer.get(node.input[1])
    if weight_dq is None or weight_dq.op_type != "DequantizeLinear":
        return False
    weight_q = producer.get(weight_dq.input[0])
    return weight_q is not None and weight_q.op_type == "QuantizeLinear"


def validate_result(
    model: ModelProto,
    before_qdq: Counter,
    before_plugins: Mapping[str, str],
) -> None:
    """严格验证只新增18个 attention q/MatMul 的完整 Q/DQ。"""
    onnx.checker.check_model(model)
    nodes_by_name, producer, consumers, initializers = build_maps(model)

    quantized_attention_matmuls = {
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and "/attn/" in node.name
        and is_quantized_weight_matmul(node, producer)
    }
    expected = set(BLOCK3_Q_TARGETS)
    missing = sorted(expected - quantized_attention_matmuls)
    unexpected = sorted(quantized_attention_matmuls - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Attention MatMul 量化集合错误:\n  missing={}\n  unexpected={}".format(
                missing,
                unexpected,
            )
        )

    for block_index, target_name in enumerate(BLOCK3_Q_TARGETS):
        target = require_node(nodes_by_name, target_name, "MatMul")
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

        if get_axis(weight_q) != 1 or get_axis(weight_dq) != 1:
            raise RuntimeError("{} 权重 Q/DQ axis 不是 1".format(target_name))

        raw_weight = weight_q.input[0]
        weight = initializer_array(initializers, raw_weight)
        scale_q = initializer_array(initializers, weight_q.input[1])
        scale_dq = initializer_array(initializers, weight_dq.input[1])
        zero_q = initializer_array(initializers, weight_q.input[2])
        zero_dq = initializer_array(initializers, weight_dq.input[2])
        if scale_q.shape != (weight.shape[1],):
            raise RuntimeError("{} 权重 Q scale shape 错误".format(target_name))
        if scale_dq.shape != scale_q.shape or not np.array_equal(scale_q, scale_dq):
            raise RuntimeError("{} 权重 Q/DQ scale 不一致".format(target_name))
        if zero_q.dtype != np.int8 or zero_dq.dtype != np.int8:
            raise RuntimeError("{} 权重 zero-point 不是 INT8".format(target_name))
        if np.any(zero_q != 0) or np.any(zero_dq != 0):
            raise RuntimeError("{} 权重 zero-point 不是全零".format(target_name))

        activation_scale_q = initializer_array(
            initializers,
            activation_q.input[1],
        )
        activation_scale_dq = initializer_array(
            initializers,
            activation_dq.input[1],
        )
        if activation_scale_q.shape != () or activation_scale_dq.shape != ():
            raise RuntimeError("{} 激活 scale 不是标量".format(target_name))
        if not np.array_equal(activation_scale_q, activation_scale_dq):
            raise RuntimeError("{} 激活 Q/DQ scale 不一致".format(target_name))

        # KV 和 projection 必须保持未量化，防止本次实验边界扩大。
        for suffix in ("kv/MatMul", "proj/MatMul"):
            other_name = "/model/block3.{}/attn/{}".format(
                block_index,
                suffix,
            )
            other = require_node(nodes_by_name, other_name, "MatMul")
            if is_quantized_weight_matmul(other, producer):
                raise RuntimeError("出现非目标 Attention 量化: {}".format(other_name))

    after_qdq = qdq_count(model)
    if after_qdq["QuantizeLinear"] - before_qdq["QuantizeLinear"] != 36:
        raise RuntimeError(
            "新增 QuantizeLinear 数量不是36: before={}, after={}".format(
                before_qdq,
                after_qdq,
            )
        )
    if after_qdq["DequantizeLinear"] - before_qdq["DequantizeLinear"] != 36:
        raise RuntimeError(
            "新增 DequantizeLinear 数量不是36: before={}, after={}".format(
                before_qdq,
                after_qdq,
            )
        )

    after_plugins = plugin_snapshot(model)
    if dict(before_plugins) != after_plugins:
        raise RuntimeError("V16 的 egcinet Plugin 节点发生变化")

    print("\n[VALIDATE]")
    print("  block3 attention q INT8 targets : 18/18")
    print("  dedicated activation Q/DQ       : 18/18")
    print("  per-channel weight Q/DQ axis=1  : 18/18")
    print("  KV/projection quantized          : 0")
    print("  added QuantizeLinear             : 36")
    print("  added DequantizeLinear           : 36")
    print("  egcinet Plugin nodes unchanged   : {}".format(len(after_plugins)))


def rewrite_model(
    model: ModelProto,
    cache_scales: Mapping[str, float],
) -> Tuple[Counter, Dict[str, str]]:
    """执行 V17 定点改图，并返回改图前的验证基线。"""
    onnx.checker.check_model(model)
    before_qdq = qdq_count(model)
    before_plugins = plugin_snapshot(model)
    nodes_by_name, producer, consumers, initializers = build_maps(model)

    # V16 中不应该已经存在 Attention 权重量化，避免把旧错误 V17 当输入。
    existing_attention_quantized = sorted(
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and "/attn/" in node.name
        and is_quantized_weight_matmul(node, producer)
    )
    if existing_attention_quantized:
        raise RuntimeError(
            "输入不是干净的 V16，已经存在 Attention 权重量化: {}".format(
                existing_attention_quantized
            )
        )

    injections: Dict[str, List[NodeProto]] = {}
    appended_initializers: List[TensorProto] = []

    for block_index, target_name in enumerate(BLOCK3_Q_TARGETS):
        target = require_node(nodes_by_name, target_name, "MatMul")
        template = inspect_fc1_template(
            block_index,
            nodes_by_name,
            producer,
            initializers,
        )
        new_nodes, new_initializers, _, _ = create_qdq_for_target(
            model,
            block_index,
            target,
            producer,
            consumers,
            initializers,
            cache_scales,
            template,
        )
        injections[target_name] = new_nodes
        appended_initializers.extend(new_initializers)

    # 按原 MatMul 的位置插入 Q/DQ，保持 ONNX 节点拓扑顺序合法。
    original_nodes = list(model.graph.node)
    rewritten_nodes: List[NodeProto] = []
    for node in original_nodes:
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
            "V17修正版：基于V16，只为18个block3 attention q/MatMul "
            "插入专用激活Q/DQ和逐输出通道权重Q/DQ"
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v16_"
            "block4_packed_dwconv_gelu.onnx"
        ),
        help="已经完成 Plugin 改图的 V16 ONNX",
    )
    parser.add_argument(
        "--cache",
        default="deploy/egcienet_352_multiclass_int8.cache",
        help="生成原模型 INT8 动态范围的 TensorRT calibration cache",
    )
    parser.add_argument(
        "--output",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v17_"
            "block3_attn_q_matmul_qdq.onnx"
        ),
        help="修正后的 V17 ONNX",
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
        raise FileNotFoundError("找不到 V16 ONNX: {}".format(input_path))
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
    print("[TARGET] block3.0~17 attention q/MatMul，共18个")
    print("[KEEP  ] SR/KV/MHA/projection/residual 和全部 Plugin 不变")

    cache_scales = load_cache_scales(cache_path)
    model = onnx.load(str(input_path))
    before_qdq, before_plugins = rewrite_model(model, cache_scales)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    # 重新加载最终文件再校验一次，避免只在内存图上通过。
    saved_model = onnx.load(str(output_path))
    validate_result(saved_model, before_qdq, before_plugins)

    print("\n[DONE] {}".format(output_path))
    print(
        "[NEXT] 构建 TensorRT 后必须确认 q/MatMul kernel 从 "
        "f16f16 变为 i8/s8 Tensor Core kernel"
    )


if __name__ == "__main__":
    main()
