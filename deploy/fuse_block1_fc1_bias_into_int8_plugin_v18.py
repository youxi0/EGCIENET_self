#!/usr/bin/env python3

"""
V18 ONNX 接口实验：让三个 block1 Packed DWConv + GELU Plugin 直接消费
fc1 MatMul 的 INT8 输出，并把原 fc1/Add 的静态 bias 移入 Plugin 属性。

改图前：

  激活 DQ --+
             +-> fc1/MatMul -> fc1/Add -> Plugin v1(FP16) -> fc2
  权重 DQ --+                    ^
                                  fc1 bias

改图后：

  激活 DQ --+
             +-> fc1/MatMul -> QuantizeLinear -> Plugin v2(INT8) -> fc2
  权重 DQ --+                              ^
                                             input_scale
                                             packed_fc1_bias

本脚本只修改 ONNX，不实现 Plugin v2。为了防止旧 Plugin v1 把 INT8 地址
误当成 FP16 地址，三个改写节点会明确设置 plugin_version="2"。在后续
C++/CUDA Creator v2 实现之前，该 ONNX 预期不能成功构建 engine。

量化边界说明：
  - fc1 MatMul 的两个输入 Q/DQ 保持不变，TensorRT 仍可选择 INT8 GEMM；
  - 新增的输出 QuantizeLinear 位于 MatMul 与 Plugin 之间；
  - 原 fc1/Add 被删除，bias 按 half2 位模式打包到 packed_fc1_bias；
  - Plugin v2 将来必须按 x_fp = input_scale * int8_x + fc1_bias 解释输入；
  - fc2、残差、Attention 和其余 Plugin 均不修改。
"""

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper


BLOCK_COUNT = 3
PLUGIN_OP = "EGCINET_Block1PackedDwconv"
PLUGIN_DOMAIN = "egcinet"
PLUGIN_V1 = "1"
PLUGIN_V2 = "2"


def fc1_matmul_name(block_index: int) -> str:
    return "/model/block1.{}/mlp/fc1/MatMul".format(block_index)


def fc1_add_name(block_index: int) -> str:
    return "/model/block1.{}/mlp/fc1/Add".format(block_index)


def plugin_v1_name(block_index: int) -> str:
    return "/model/block1.{}/mlp/Block1PackedDwconvGelu".format(block_index)


def plugin_v2_name(block_index: int) -> str:
    return "/model/block1.{}/mlp/Block1Int8PackedDwconvGelu".format(
        block_index
    )


def output_q_prefix(block_index: int) -> str:
    return "/model/block1.{}/mlp/fc1/MatMul_output_V18".format(block_index)


def build_maps(
    model: ModelProto,
) -> Tuple[
    Dict[str, NodeProto],
    Dict[str, NodeProto],
    Dict[str, List[NodeProto]],
    Dict[str, TensorProto],
]:
    """建立节点、tensor 生产者、消费者和 initializer 索引。"""
    nodes_by_name: Dict[str, NodeProto] = {}
    producer: Dict[str, NodeProto] = {}
    consumers: Dict[str, List[NodeProto]] = {}

    for node in model.graph.node:
        if not node.name:
            raise RuntimeError("图中存在无名称节点，无法安全做定点改图")
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
    domain: str = "",
) -> NodeProto:
    """严格按名称、算子类型和 domain 取得节点。"""
    node = nodes_by_name.get(name)
    if node is None:
        raise RuntimeError("找不到节点: {}".format(name))
    if node.op_type != op_type or node.domain != domain:
        raise RuntimeError(
            "{} 类型错误，期望 {}::{}, 实际 {}::{}".format(
                name,
                domain,
                op_type,
                node.domain,
                node.op_type,
            )
        )
    return node


def node_attributes(node: NodeProto) -> Dict[str, object]:
    """把 ONNX AttributeProto 转成便于校验的字典。"""
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def replace_attribute(node: NodeProto, name: str, value: object) -> None:
    """替换或新增一个节点属性，保证同名属性只保留一份。"""
    kept = [
        attribute
        for attribute in node.attribute
        if attribute.name != name
    ]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def hash_node(node: NodeProto) -> str:
    """对节点完整序列化内容做摘要。"""
    return hashlib.sha256(node.SerializeToString()).hexdigest()


def qdq_count(model: ModelProto) -> Counter:
    """统计图中的显式 Q/DQ 数量。"""
    return Counter(
        node.op_type
        for node in model.graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
    )


def half_pairs_to_signed_words(values: np.ndarray) -> List[int]:
    """把最后一维为2的 FP16 数组按 half2 位模式打包成有符号 int32。"""
    pairs = np.ascontiguousarray(values, dtype=np.float16)
    if pairs.ndim < 1 or pairs.shape[-1] != 2:
        raise ValueError("half2 打包要求最后一维等于2")

    bits = pairs.view(np.uint16)
    words = (
        bits[..., 0].astype(np.uint32)
        | (bits[..., 1].astype(np.uint32) << np.uint32(16))
    )
    return np.ascontiguousarray(words).view(np.int32).reshape(-1).tolist()


def pack_fc1_bias(bias: np.ndarray, channels: int, node_name: str) -> List[int]:
    """把 fc1 的 FP16[C] bias 打包成 Plugin 使用的 half2[C/2]。"""
    bias = np.asarray(bias)
    if channels <= 0 or (channels & 1) != 0:
        raise RuntimeError("{} channels 必须是正偶数".format(node_name))
    if bias.size != channels:
        raise RuntimeError(
            "{} bias 大小错误，期望{}，实际{}".format(
                node_name,
                channels,
                bias.size,
            )
        )

    pairs = np.ascontiguousarray(
        bias,
        dtype=np.float16,
    ).reshape(channels // 2, 2)
    packed = half_pairs_to_signed_words(pairs)
    if len(packed) != channels // 2:
        raise RuntimeError("{} half2 bias 打包长度错误".format(node_name))
    return packed


def load_cache_scales(cache_path: Path) -> Dict[str, float]:
    """
    只使用 ModelOpt 读取 TensorRT calibration cache，不让 ModelOpt 改图。
    """
    try:
        from modelopt.onnx.quantization.calib_utils import (
            import_scales_from_calib_cache,
        )
    except ImportError as error:
        raise ImportError(
            "请在已安装 nvidia-modelopt[onnx] 的环境中运行本脚本"
        ) from error

    raw_scales = import_scales_from_calib_cache(str(cache_path))
    scales = {
        str(key): float(value)
        for key, value in raw_scales.items()
    }
    if not scales:
        raise RuntimeError("calibration cache 中没有 scale")
    return scales


def find_matmul_output_scale(
    tensor_name: str,
    cache_scales: Mapping[str, float],
) -> Tuple[str, float]:
    """
    查找 fc1 MatMul 的输出 scale。

    注意：fc1/Add 已经移入 Plugin，所以必须校准 MatMul 的前 bias 输出，
    不能回退使用 Add 输出的 scale。若 cache 不含该 tensor，本脚本直接失败。
    """
    candidates = [tensor_name + "_scale", tensor_name]
    for key in candidates:
        if key not in cache_scales:
            continue
        scale = float(cache_scales[key])
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("scale 非法: {}={}".format(key, scale))
        return key, scale

    tail = tensor_name.split("/")[-3:]
    keyword = "/".join(tail)
    nearby = sorted(
        key
        for key in cache_scales
        if keyword in key or tensor_name in key
    )[:20]
    raise RuntimeError(
        "找不到 fc1 MatMul 输出 calibration scale:\n"
        "  tensor : {}\n"
        "  tried  : {}\n"
        "  nearby : {}\n"
        "不能改用 fc1/Add 输出 scale，因为 bias 已经移动到 Plugin 内。".format(
            tensor_name,
            candidates,
            nearby,
        )
    )


def require_available_names(
    model: ModelProto,
    node_names: Iterable[str],
    tensor_names: Iterable[str],
) -> None:
    """检查新增的节点名和 tensor 名不会与 V16 冲突。"""
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
            "V18 名称冲突:\n  nodes={}\n  tensors={}".format(
                node_collisions,
                tensor_collisions,
            )
        )


def scalar_template_for_block1_fc1(
    matmul: NodeProto,
    producer: Mapping[str, NodeProto],
    initializers: Mapping[str, TensorProto],
) -> Tuple[TensorProto, TensorProto]:
    """
    从已经生效的 block1 fc1 输入 Q/DQ 取得 FP16 scale 和 INT8 零点模板。
    """
    if len(matmul.input) != 2:
        raise RuntimeError("{} 输入数量不是2".format(matmul.name))
    activation_dq = producer.get(matmul.input[0])
    if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
        raise RuntimeError("{} 激活输入不是已有 DQ".format(matmul.name))
    activation_q = producer.get(activation_dq.input[0])
    if activation_q is None or activation_q.op_type != "QuantizeLinear":
        raise RuntimeError("{} 前面缺少已有激活 Q".format(matmul.name))
    if len(activation_q.input) != 3:
        raise RuntimeError("{} 激活 Q 缺少 scale/zero-point".format(matmul.name))

    scale_template = initializers.get(activation_q.input[1])
    zero_template = initializers.get(activation_q.input[2])
    if scale_template is None or zero_template is None:
        raise RuntimeError("{} 激活 Q initializer 不完整".format(matmul.name))

    scale_array = np.asarray(numpy_helper.to_array(scale_template))
    zero_array = np.asarray(numpy_helper.to_array(zero_template))
    if scale_array.shape != ():
        raise RuntimeError("{} 激活 scale 不是标量".format(matmul.name))
    if zero_array.shape != () or zero_array.dtype != np.int8:
        raise RuntimeError("{} 激活 zero-point 不是 INT8 标量".format(matmul.name))
    if int(zero_array) != 0:
        raise RuntimeError("{} 激活 zero-point 不是对称零点".format(matmul.name))
    return scale_template, zero_template


def make_scalar_like(
    template: TensorProto,
    name: str,
    value: object,
) -> TensorProto:
    """沿用现有 Q/DQ 标量 initializer 的 dtype。"""
    dtype = numpy_helper.to_array(template).dtype
    array = np.asarray(value, dtype=dtype).reshape(())
    return numpy_helper.from_array(array, name=name)


def get_plugin_version(plugin: NodeProto) -> str:
    """读取自定义 Plugin 的 plugin_version 字符串属性。"""
    value = node_attributes(plugin).get("plugin_version")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise RuntimeError("{} 缺少合法 plugin_version".format(plugin.name))


def rewrite_one_block(
    model: ModelProto,
    block_index: int,
    cache_scales: Mapping[str, float],
    nodes_by_name: Mapping[str, NodeProto],
    producer: Mapping[str, NodeProto],
    consumers: Mapping[str, Sequence[NodeProto]],
    initializers: Mapping[str, TensorProto],
) -> Dict[str, object]:
    """把一个 block1 改成 MatMul -> Q(INT8) -> Plugin v2。"""
    matmul = require_node(
        nodes_by_name,
        fc1_matmul_name(block_index),
        "MatMul",
    )
    add = require_node(
        nodes_by_name,
        fc1_add_name(block_index),
        "Add",
    )
    plugin = require_node(
        nodes_by_name,
        plugin_v1_name(block_index),
        PLUGIN_OP,
        PLUGIN_DOMAIN,
    )

    if get_plugin_version(plugin) != PLUGIN_V1:
        raise RuntimeError("{} 不是 Plugin v1 输入图".format(plugin.name))
    if len(matmul.output) != 1 or len(add.input) != 2 or len(add.output) != 1:
        raise RuntimeError("block1.{} fc1 接口异常".format(block_index))
    if len(plugin.input) != 1 or len(plugin.output) != 1:
        raise RuntimeError("{} 必须是单输入单输出".format(plugin.name))

    matmul_output = matmul.output[0]
    add_output = add.output[0]
    if matmul_output not in add.input:
        raise RuntimeError("{} 不消费 fc1 MatMul 输出".format(add.name))
    if plugin.input[0] != add_output:
        raise RuntimeError("{} 输入不是 fc1/Add 输出".format(plugin.name))

    matmul_consumers = list(consumers.get(matmul_output, []))
    add_consumers = list(consumers.get(add_output, []))
    if len(matmul_consumers) != 1 or matmul_consumers[0].name != add.name:
        raise RuntimeError(
            "{} 输出消费者异常: {}".format(
                matmul.name,
                [node.name for node in matmul_consumers],
            )
        )
    if len(add_consumers) != 1 or add_consumers[0].name != plugin.name:
        raise RuntimeError(
            "{} 输出消费者异常: {}".format(
                add.name,
                [node.name for node in add_consumers],
            )
        )

    bias_inputs = [name for name in add.input if name in initializers]
    if len(bias_inputs) != 1:
        raise RuntimeError("{} 没有唯一静态 bias".format(add.name))
    bias_name = bias_inputs[0]
    bias_consumers = list(consumers.get(bias_name, []))
    if len(bias_consumers) != 1 or bias_consumers[0].name != add.name:
        raise RuntimeError("{} 不是 fc1/Add 专用 bias".format(bias_name))

    attributes = node_attributes(plugin)
    channels = int(attributes.get("channels", 0))
    if channels != 256:
        raise RuntimeError(
            "{} channels 不是 block1 固定值256".format(plugin.name)
        )
    fc1_bias = np.asarray(numpy_helper.to_array(initializers[bias_name]))
    packed_fc1_bias = pack_fc1_bias(fc1_bias, channels, add.name)

    cache_key, cache_scale = find_matmul_output_scale(
        matmul_output,
        cache_scales,
    )
    scale_template, zero_template = scalar_template_for_block1_fc1(
        matmul,
        producer,
        initializers,
    )

    # 与现有 ModelOpt Q/DQ 一样使用 FP16 scale。Plugin 属性保存的必须是
    # 同一个 FP16 舍入值，否则 Plugin 反量化与 QuantizeLinear 不一致。
    scale_dtype = numpy_helper.to_array(scale_template).dtype
    rounded_scale = float(np.asarray(cache_scale, dtype=scale_dtype))
    if not np.isfinite(rounded_scale) or rounded_scale <= 0.0:
        raise RuntimeError(
            "{} scale 转成 {} 后非法: {}".format(
                matmul.name,
                scale_dtype,
                rounded_scale,
            )
        )

    prefix = output_q_prefix(block_index)
    q_name = prefix + "_QuantizeLinear"
    q_output = prefix + "_QuantizeLinear_Output"
    scale_name = prefix + "_scale"
    zero_name = prefix + "_zero_point"
    new_plugin_name = plugin_v2_name(block_index)

    require_available_names(
        model,
        [q_name, new_plugin_name],
        [q_output, scale_name, zero_name],
    )

    scale_initializer = make_scalar_like(
        scale_template,
        scale_name,
        rounded_scale,
    )
    zero_initializer = make_scalar_like(
        zero_template,
        zero_name,
        0,
    )
    output_q = helper.make_node(
        "QuantizeLinear",
        inputs=[matmul_output, scale_name, zero_name],
        outputs=[q_output],
        name=q_name,
    )

    # Plugin v2 的 ONNX 契约：INT8 LINEAR 输入、FP16 LINEAR 输出；内部先按
    # input_scale 反量化并加 packed_fc1_bias，再执行原 Packed DWConv+GELU。
    old_plugin_name = plugin.name
    plugin.name = new_plugin_name
    plugin.input[0] = q_output
    replace_attribute(plugin, "plugin_version", PLUGIN_V2)
    replace_attribute(plugin, "input_scale", rounded_scale)
    replace_attribute(plugin, "input_zero_point", 0)
    replace_attribute(plugin, "fuse_fc1_bias", 1)
    replace_attribute(plugin, "packed_fc1_bias", packed_fc1_bias)

    print(
        "[REWRITE] block1.{}: remove fc1/Add, MatMul output -> INT8 Q -> "
        "Plugin v2; scale={:.9g}, cache_key={}, packed_fc1_bias={}".format(
            block_index,
            rounded_scale,
            cache_key,
            len(packed_fc1_bias),
        )
    )
    return {
        "block_index": block_index,
        "matmul_name": matmul.name,
        "matmul_output": matmul_output,
        "add_name": add.name,
        "add_output": add_output,
        "old_plugin_name": old_plugin_name,
        "plugin_name": new_plugin_name,
        "bias_name": bias_name,
        "q_name": q_name,
        "q_output": q_output,
        "scale_name": scale_name,
        "zero_name": zero_name,
        "rounded_scale": rounded_scale,
        "packed_fc1_bias": packed_fc1_bias,
        "q_node": output_q,
        "scale_initializer": scale_initializer,
        "zero_initializer": zero_initializer,
    }


def protected_node_snapshot(model: ModelProto) -> Dict[str, str]:
    """记录除3个目标 Add/Plugin 外的所有节点，验证改图范围没有扩大。"""
    mutable_names = {
        name
        for block_index in range(BLOCK_COUNT)
        for name in (
            fc1_add_name(block_index),
            plugin_v1_name(block_index),
            plugin_v2_name(block_index),
        )
    }
    return {
        node.name: hash_node(node)
        for node in model.graph.node
        if node.name not in mutable_names
        and not node.name.endswith("_V18_QuantizeLinear")
    }


def validate_result(
    model: ModelProto,
    before_qdq: Counter,
    protected_before: Mapping[str, str],
    records: Sequence[Mapping[str, object]],
) -> None:
    """严格检查3个 INT8 Plugin 输入边界及非目标图不变。"""
    onnx.checker.check_model(model)
    nodes_by_name, producer, consumers, initializers = build_maps(model)

    for record in records:
        block_index = int(record["block_index"])
        matmul = require_node(
            nodes_by_name,
            str(record["matmul_name"]),
            "MatMul",
        )
        q = require_node(
            nodes_by_name,
            str(record["q_name"]),
            "QuantizeLinear",
        )
        plugin = require_node(
            nodes_by_name,
            str(record["plugin_name"]),
            PLUGIN_OP,
            PLUGIN_DOMAIN,
        )

        if str(record["add_name"]) in nodes_by_name:
            raise RuntimeError("block1.{} fc1/Add 仍在图中".format(block_index))
        if str(record["old_plugin_name"]) in nodes_by_name:
            raise RuntimeError("block1.{} Plugin v1 节点名仍在图中".format(block_index))
        if str(record["bias_name"]) in initializers:
            raise RuntimeError("block1.{} fc1 bias initializer 未移除".format(block_index))

        if list(q.input) != [
            matmul.output[0],
            str(record["scale_name"]),
            str(record["zero_name"]),
        ]:
            raise RuntimeError("block1.{} 输出 Q 输入错误".format(block_index))
        if plugin.input[0] != q.output[0]:
            raise RuntimeError("block1.{} Plugin 不直接消费 INT8 Q 输出".format(block_index))

        matmul_output_consumers = consumers.get(matmul.output[0], [])
        q_output_consumers = consumers.get(q.output[0], [])
        if len(matmul_output_consumers) != 1 or matmul_output_consumers[0].name != q.name:
            raise RuntimeError("block1.{} MatMul 输出不是专用 Q".format(block_index))
        if len(q_output_consumers) != 1 or q_output_consumers[0].name != plugin.name:
            raise RuntimeError("block1.{} INT8 Q 输出不是专用 Plugin 输入".format(block_index))

        scale = np.asarray(
            numpy_helper.to_array(initializers[str(record["scale_name"])]),
        )
        zero = np.asarray(
            numpy_helper.to_array(initializers[str(record["zero_name"])]),
        )
        if scale.shape != () or scale.dtype != np.float16:
            raise RuntimeError("block1.{} 输出 Q scale 不是 FP16 标量".format(block_index))
        if zero.shape != () or zero.dtype != np.int8 or int(zero) != 0:
            raise RuntimeError("block1.{} 输出 Q zero-point 错误".format(block_index))

        attributes = node_attributes(plugin)
        version = get_plugin_version(plugin)
        if version != PLUGIN_V2:
            raise RuntimeError("block1.{} Plugin 不是 v2".format(block_index))
        if int(attributes.get("input_zero_point", -1)) != 0:
            raise RuntimeError("block1.{} Plugin zero-point 不是0".format(block_index))
        if int(attributes.get("fuse_fc1_bias", 0)) != 1:
            raise RuntimeError("block1.{} 未声明融合 fc1 bias".format(block_index))
        plugin_scale = float(attributes.get("input_scale", 0.0))
        if plugin_scale != float(scale):
            raise RuntimeError("block1.{} Q scale 与 Plugin scale 不一致".format(block_index))
        packed_bias = list(attributes.get("packed_fc1_bias", []))
        if packed_bias != list(record["packed_fc1_bias"]):
            raise RuntimeError("block1.{} packed_fc1_bias 不一致".format(block_index))
        if len(packed_bias) != 128:
            raise RuntimeError("block1.{} packed_fc1_bias 长度不是128".format(block_index))

        # fc1 的两个已有 DQ 输入必须继续保留，确保 GEMM 量化边界没有被破坏。
        for input_name in matmul.input:
            parent = producer.get(input_name)
            if parent is None or parent.op_type != "DequantizeLinear":
                raise RuntimeError("block1.{} fc1 MatMul 输入不再是 DQ".format(block_index))

    after_qdq = qdq_count(model)
    if after_qdq["QuantizeLinear"] - before_qdq["QuantizeLinear"] != 3:
        raise RuntimeError(
            "新增 QuantizeLinear 数量不是3: before={}, after={}".format(
                before_qdq,
                after_qdq,
            )
        )
    if after_qdq["DequantizeLinear"] != before_qdq["DequantizeLinear"]:
        raise RuntimeError("本次改图不应新增或删除 DequantizeLinear")

    protected_after = protected_node_snapshot(model)
    for name, digest in protected_before.items():
        if protected_after.get(name) != digest:
            raise RuntimeError("非目标节点发生变化: {}".format(name))

    plugin_v2_nodes = [
        node
        for node in model.graph.node
        if node.domain == PLUGIN_DOMAIN
        and node.op_type == PLUGIN_OP
        and get_plugin_version(node) == PLUGIN_V2
    ]
    if len(plugin_v2_nodes) != BLOCK_COUNT:
        raise RuntimeError("Plugin v2 节点数量不是3")

    print("\n[VALIDATE]")
    print("  block1 fc1 MatMul output Q       : 3/3")
    print("  block1 Plugin v2 INT8 input      : 3/3")
    print("  fused fc1/Add nodes removed      : 3/3")
    print("  packed fc1 bias half2 words      : 128 x 3")
    print("  added QuantizeLinear             : 3")
    print("  added DequantizeLinear           : 0")
    print("  fc1 input Q/DQ preserved         : 3/3")
    print("  protected non-target nodes       : unchanged")


def rewrite_model(
    model: ModelProto,
    cache_scales: Mapping[str, float],
) -> Tuple[Counter, Dict[str, str], List[Dict[str, object]]]:
    """执行3个 block1 的 V18 ONNX 接口改写。"""
    onnx.checker.check_model(model)
    before_qdq = qdq_count(model)
    protected_before = protected_node_snapshot(model)
    nodes_by_name, producer, consumers, initializers = build_maps(model)

    records: List[Dict[str, object]] = []
    for block_index in range(BLOCK_COUNT):
        records.append(
            rewrite_one_block(
                model,
                block_index,
                cache_scales,
                nodes_by_name,
                producer,
                consumers,
                initializers,
            )
        )

    q_by_add_name = {
        str(record["add_name"]): record["q_node"]
        for record in records
    }
    remove_add_names = set(q_by_add_name)
    remove_bias_names = {
        str(record["bias_name"])
        for record in records
    }

    # 在原 Add 的位置插入 QuantizeLinear，确保 MatMul 已经在它之前产生输出。
    rewritten_nodes: List[NodeProto] = []
    for node in model.graph.node:
        if node.name in remove_add_names:
            rewritten_nodes.append(q_by_add_name[node.name])
            continue
        rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in remove_bias_names
    ]
    for record in records:
        kept_initializers.append(record["scale_initializer"])
        kept_initializers.append(record["zero_initializer"])
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    validate_result(model, before_qdq, protected_before, records)
    return before_qdq, protected_before, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V18：删除3个block1 fc1/Add，把bias移入Plugin v2，并让Plugin "
            "直接消费fc1 MatMul输出的INT8 QuantizeLinear tensor"
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v16_"
            "block4_packed_dwconv_gelu.onnx"
        ),
        help="含有3个 block1 PackedDwconvGelu Plugin v1 的输入 ONNX",
    )
    parser.add_argument(
        "--cache",
        default="deploy/egcienet_352_multiclass_int8.cache",
        help="包含 fc1 MatMul 输出动态范围的 TensorRT calibration cache",
    )
    parser.add_argument(
        "--output",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v18_"
            "block1_int8_plugin_input.onnx"
        ),
        help="Plugin v2 INT8 输入接口 ONNX",
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
        raise FileNotFoundError("找不到输入 ONNX: {}".format(input_path))
    if not cache_path.is_file():
        raise FileNotFoundError("找不到 calibration cache: {}".format(cache_path))
    if input_path.resolve() == output_path.resolve():
        raise RuntimeError("输出不能覆盖输入 ONNX")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "输出已经存在: {}；如需覆盖请增加 --force".format(output_path)
        )

    print("[INPUT ] {}".format(input_path))
    print("[CACHE ] {}".format(cache_path))
    print("[OUTPUT] {}".format(output_path))
    print("[TARGET] 3 x block1 fc1 MatMul output Q -> Plugin v2 INT8 input")
    print("[FUSE  ] remove fc1/Add; pack fc1 bias into Plugin attributes")
    print("[NOTE  ] this turn only defines ONNX; Plugin Creator v2 is not implemented")

    cache_scales = load_cache_scales(cache_path)
    model = onnx.load(str(input_path))
    before_qdq, protected_before, records = rewrite_model(model, cache_scales)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    # 重新加载落盘文件验证，避免只在内存 protobuf 上通过。
    saved_model = onnx.load(str(output_path))
    validate_result(
        saved_model,
        before_qdq,
        protected_before,
        records,
    )

    print("\n[DONE] {}".format(output_path))
    print(
        "[BLOCKED-BY-DESIGN] 需要实现 {} version={} 后才能构建 engine".format(
            PLUGIN_OP,
            PLUGIN_V2,
        )
    )


if __name__ == "__main__":
    main()
