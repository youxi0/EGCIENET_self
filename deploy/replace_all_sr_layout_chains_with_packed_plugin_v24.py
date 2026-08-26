#!/usr/bin/env python3

"""
V24：以 V22 为量化基线，把 block1～3 的全部25条 SR 布局链替换为 Plugin。

原始链：

  token [B,H*W,C]
    -> Transpose_1 [B,C,H*W]
    -> Reshape_1   [B,C,H,W]
    -> sr/Conv     [B,C,11,11]
    -> Reshape_2   [B,C,121]
    -> Transpose_2 [B,121,C]

替换后：

  token [B,H*W,C]
    -> EGCINET_FusedSpatialReduction
    -> token [B,121,C]

Plugin 输入契约：
  input0：FP16 token-major 激活 [B,H*W,C]
  input1：FP16 packed weight [K,C_out]，K=kh*kw*C_in
  input2：FP16 bias [C_out]

权重从 ONNX Conv 的 OIHW 排列：

  W[co,ci,kh,kw]

预排为适合直接读取 token patch 的连续排列：

  packed[kh,kw,ci,co] -> [K,C_out]

这样 Plugin kernel 可以按 patch 像素、输入通道、输出通道的顺序读取，且
输出通道连续，后续实现可直接用 half2/Tensor Core tile。脚本只定义 ONNX
图和权重物理契约，不包含任何 TensorRT Plugin/CUDA 实现。

注意：block4 的 sr_ratio=1，真实图中没有 SR Conv，因此 V24 明确要求
block4 的 SR Plugin 数量为0，不创建无意义的 Identity Plugin。
"""

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper

import insert_attn_q_block134_kv_all_qdq_v22 as v22


v20 = v22.v20
v17 = v22.v17

PLUGIN_OP = "EGCINET_FusedSpatialReduction"
PLUGIN_DOMAIN = "egcinet"
PLUGIN_NAMESPACE = ""
PLUGIN_VERSION = "1"
PACKED_WEIGHT_LAYOUT = "KHKW_CI_CO_ROW_MAJOR_FP16"

# stage -> (层数, 输入高, 输入宽, 通道数, SR kernel/stride)
STAGE_SPECS: Dict[int, Tuple[int, int, int, int, int]] = {
    1: (3, 88, 88, 64, 8),
    2: (4, 44, 44, 128, 4),
    3: (18, 22, 22, 320, 2),
}


def node_attributes(node: NodeProto) -> Dict[str, object]:
    """把 ONNX attribute 转成便于严格检查的字典。"""
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def require_node(
    nodes_by_name: Mapping[str, NodeProto],
    name: str,
    op_type: str,
) -> NodeProto:
    """按完整名称和类型精确定位节点。"""
    node = nodes_by_name.get(name)
    if node is None:
        raise RuntimeError("找不到节点: {}".format(name))
    if node.op_type != op_type:
        raise RuntimeError(
            "{} 类型错误: expected={}, actual={}".format(
                name,
                op_type,
                node.op_type,
            )
        )
    return node


def constant_array(node: NodeProto) -> np.ndarray:
    """读取 Constant 节点的 value tensor。"""
    if node.op_type != "Constant":
        raise RuntimeError("{} 不是 Constant".format(node.name))
    attributes = node_attributes(node)
    value = attributes.get("value")
    if not isinstance(value, TensorProto):
        raise RuntimeError("{} 缺少 tensor value".format(node.name))
    return np.asarray(numpy_helper.to_array(value))


def require_only_consumers(
    tensor_name: str,
    consumers: Mapping[str, Sequence[NodeProto]],
    expected_names: Set[str],
) -> None:
    """确认 tensor 没有链外消费者，避免删除共享节点。"""
    actual_names = {node.name for node in consumers.get(tensor_name, [])}
    if actual_names != expected_names:
        raise RuntimeError(
            "{} 消费者错误: expected={}, actual={}".format(
                tensor_name,
                sorted(expected_names),
                sorted(actual_names),
            )
        )


def array_sha256(array: np.ndarray) -> str:
    """计算连续数组内容摘要，用于保存/重载后检查 packed weight。"""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def node_sha256(node: NodeProto) -> str:
    """记录非目标节点序列化摘要，防止改图污染注意力或既有 Plugin。"""
    return hashlib.sha256(node.SerializeToString()).hexdigest()


def ensure_plugin_opset(model: ModelProto) -> None:
    """确保 egcinet 自定义 domain 已声明为 opset 1。"""
    for opset in model.opset_import:
        if opset.domain != PLUGIN_DOMAIN:
            continue
        if opset.version != 1:
            raise RuntimeError(
                "{} domain opset 不是1: {}".format(
                    PLUGIN_DOMAIN,
                    opset.version,
                )
            )
        return
    model.opset_import.append(helper.make_operatorsetid(PLUGIN_DOMAIN, 1))


def prepare_replacement(
    model: ModelProto,
    stage: int,
    layer_index: int,
    nodes_by_name: Mapping[str, NodeProto],
    consumers: Mapping[str, Sequence[NodeProto]],
    initializers: Mapping[str, TensorProto],
    node_order: Mapping[str, int],
) -> Dict[str, object]:
    """验证一条 SR 链，创建 packed initializer 和 Plugin 节点。"""
    depth, height, width, channels, sr_ratio = STAGE_SPECS[stage]
    if layer_index < 0 or layer_index >= depth:
        raise RuntimeError("block{}.{} 超出目标范围".format(stage, layer_index))

    prefix = "/model/block{}.{}/attn".format(stage, layer_index)
    transpose_in = require_node(nodes_by_name, prefix + "/Transpose_1", "Transpose")
    constant_in = require_node(nodes_by_name, prefix + "/Constant_1", "Constant")
    reshape_in = require_node(nodes_by_name, prefix + "/Reshape_1", "Reshape")
    conv = require_node(nodes_by_name, prefix + "/sr/Conv", "Conv")
    constant_out = require_node(nodes_by_name, prefix + "/Constant_2", "Constant")
    reshape_out = require_node(nodes_by_name, prefix + "/Reshape_2", "Reshape")
    transpose_out = require_node(nodes_by_name, prefix + "/Transpose_2", "Transpose")

    chain = [
        transpose_in,
        constant_in,
        reshape_in,
        conv,
        constant_out,
        reshape_out,
        transpose_out,
    ]
    if any(len(node.output) != 1 for node in chain):
        raise RuntimeError("{} SR 链存在多输出节点".format(prefix))
    if len(transpose_in.input) != 1 or len(conv.input) != 3:
        raise RuntimeError("{} SR 输入或 Conv 输入数量错误".format(prefix))

    # 步骤1：严格确认布局、空间尺寸以及 kernel=stride=sr_ratio。
    if node_attributes(transpose_in).get("perm") != [0, 2, 1]:
        raise RuntimeError("{} 输入 Transpose perm 错误".format(prefix))
    if node_attributes(transpose_out).get("perm") != [0, 2, 1]:
        raise RuntimeError("{} 输出 Transpose perm 错误".format(prefix))
    expected_input_shape = np.asarray([1, channels, height, width], dtype=np.int64)
    expected_output_shape = np.asarray([1, channels, -1], dtype=np.int64)
    if not np.array_equal(constant_array(constant_in), expected_input_shape):
        raise RuntimeError("{} Reshape_1 shape 错误".format(prefix))
    if not np.array_equal(constant_array(constant_out), expected_output_shape):
        raise RuntimeError("{} Reshape_2 shape 错误".format(prefix))

    conv_attributes = node_attributes(conv)
    expected_conv_attributes = {
        "dilations": [1, 1],
        "group": 1,
        "kernel_shape": [sr_ratio, sr_ratio],
        "pads": [0, 0, 0, 0],
        "strides": [sr_ratio, sr_ratio],
    }
    for key, expected_value in expected_conv_attributes.items():
        if conv_attributes.get(key) != expected_value:
            raise RuntimeError(
                "{} Conv {} 错误: expected={}, actual={}".format(
                    prefix,
                    key,
                    expected_value,
                    conv_attributes.get(key),
                )
            )

    # 步骤2：链内所有中间 tensor 必须是专用边，两个 reshape shape 也不能共享。
    require_only_consumers(
        transpose_in.output[0],
        consumers,
        {reshape_in.name},
    )
    require_only_consumers(
        constant_in.output[0],
        consumers,
        {reshape_in.name},
    )
    require_only_consumers(
        reshape_in.output[0],
        consumers,
        {conv.name},
    )
    require_only_consumers(
        conv.output[0],
        consumers,
        {reshape_out.name},
    )
    require_only_consumers(
        constant_out.output[0],
        consumers,
        {reshape_out.name},
    )
    require_only_consumers(
        reshape_out.output[0],
        consumers,
        {transpose_out.name},
    )
    expected_norm_consumers = {
        prefix + "/norm/ReduceMean",
        prefix + "/norm/Sub",
    }
    require_only_consumers(
        transpose_out.output[0],
        consumers,
        expected_norm_consumers,
    )

    # 步骤3：读取 FP16 OIHW 权重，并预排为 [kh,kw,ci,co] -> [K,C_out]。
    weight_name = conv.input[1]
    bias_name = conv.input[2]
    weight_initializer = initializers.get(weight_name)
    bias_initializer = initializers.get(bias_name)
    if weight_initializer is None or bias_initializer is None:
        raise RuntimeError("{} 缺少 SR weight/bias initializer".format(prefix))
    require_only_consumers(weight_name, consumers, {conv.name})
    require_only_consumers(bias_name, consumers, {conv.name})

    weight = np.asarray(numpy_helper.to_array(weight_initializer))
    bias = np.asarray(numpy_helper.to_array(bias_initializer))
    expected_weight_shape = (channels, channels, sr_ratio, sr_ratio)
    if weight.dtype != np.float16 or weight.shape != expected_weight_shape:
        raise RuntimeError(
            "{} SR weight错误: dtype={}, shape={}".format(
                prefix,
                weight.dtype,
                weight.shape,
            )
        )
    if bias.dtype != np.float16 or bias.shape != (channels,):
        raise RuntimeError(
            "{} SR bias错误: dtype={}, shape={}".format(
                prefix,
                bias.dtype,
                bias.shape,
            )
        )

    packed_weight = np.ascontiguousarray(
        weight.transpose(2, 3, 1, 0).reshape(
            sr_ratio * sr_ratio * channels,
            channels,
        )
    )
    packed_bias = np.ascontiguousarray(bias)
    packed_weight_name = weight_name + ".V24_KHKWCI_CO_FP16"
    packed_bias_name = bias_name + ".V24_CO_FP16"
    if packed_weight_name in initializers or packed_bias_name in initializers:
        raise RuntimeError("{} packed initializer 名称冲突".format(prefix))

    packed_weight_initializer = numpy_helper.from_array(
        packed_weight,
        name=packed_weight_name,
    )
    packed_bias_initializer = numpy_helper.from_array(
        packed_bias,
        name=packed_bias_name,
    )

    # 步骤4：Plugin 直接复用原 Transpose_2 tensor 名，使 LayerNorm 以后全图不变。
    activation = transpose_in.input[0]
    output = transpose_out.output[0]
    output_height = height // sr_ratio
    output_width = width // sr_ratio
    if output_height != 11 or output_width != 11:
        raise RuntimeError("{} SR 输出不是11x11".format(prefix))

    plugin_name = prefix + "/FusedSpatialReduction"
    if plugin_name in nodes_by_name:
        raise RuntimeError("Plugin 节点名称冲突: {}".format(plugin_name))
    plugin = helper.make_node(
        PLUGIN_OP,
        inputs=[activation, packed_weight_name, packed_bias_name],
        outputs=[output],
        name=plugin_name,
        domain=PLUGIN_DOMAIN,
        plugin_namespace=PLUGIN_NAMESPACE,
        plugin_version=PLUGIN_VERSION,
        stage=stage,
        layer_index=layer_index,
        input_height=height,
        input_width=width,
        input_channels=channels,
        output_height=output_height,
        output_width=output_width,
        output_channels=channels,
        kernel_height=sr_ratio,
        kernel_width=sr_ratio,
        stride_height=sr_ratio,
        stride_width=sr_ratio,
        groups=1,
        packed_k=int(packed_weight.shape[0]),
        packed_n=int(packed_weight.shape[1]),
        input_layout="BNC_FP16",
        output_layout="BNC_FP16",
        packed_weight_layout=PACKED_WEIGHT_LAYOUT,
        fuse_layernorm=0,
    )

    return {
        "stage": stage,
        "layer_index": layer_index,
        "prefix": prefix,
        "plugin": plugin,
        "plugin_name": plugin_name,
        # shape Constant 可能因 ONNX 导出顺序出现在激活生产者之前，不能取整条链
        # 的最小序号。Plugin 复用 Transpose_1 的激活输入，因此放回 Transpose_1
        # 的原位置，既保持语义位置一致，也保证 ONNX 拓扑顺序合法。
        "insert_at": node_order[transpose_in.name],
        "remove_node_names": {node.name for node in chain},
        "remove_tensor_names": {
            tensor_name
            for node in chain
            for tensor_name in node.output
            if tensor_name and tensor_name != output
        },
        "old_weight_name": weight_name,
        "old_bias_name": bias_name,
        "packed_weight_name": packed_weight_name,
        "packed_bias_name": packed_bias_name,
        "packed_weight_initializer": packed_weight_initializer,
        "packed_bias_initializer": packed_bias_initializer,
        "packed_weight_shape": tuple(packed_weight.shape),
        "packed_bias_shape": tuple(packed_bias.shape),
        "packed_weight_sha256": array_sha256(packed_weight),
        "packed_bias_sha256": array_sha256(packed_bias),
        "activation": activation,
        "output": output,
        "height": height,
        "width": width,
        "channels": channels,
        "sr_ratio": sr_ratio,
    }


def validate_result(
    model: ModelProto,
    records: Sequence[Mapping[str, object]],
    before_qdq: Mapping[str, int],
    original_plugin_snapshot: Mapping[str, str],
    preserved_node_snapshot: Mapping[str, str],
) -> None:
    """验证25个 SR Plugin、packed initializer 和全部非目标节点。"""
    onnx.checker.check_model(model)
    nodes_by_name, producer, consumers, initializers = v17.build_maps(model)

    expected_plugin_names = {str(record["plugin_name"]) for record in records}
    actual_plugin_names = {
        node.name
        for node in model.graph.node
        if node.domain == PLUGIN_DOMAIN and node.op_type == PLUGIN_OP
    }
    if actual_plugin_names != expected_plugin_names:
        raise RuntimeError(
            "SR Plugin集合错误: expected={}, actual={}".format(
                sorted(expected_plugin_names),
                sorted(actual_plugin_names),
            )
        )

    stage_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for record in records:
        stage = int(record["stage"])
        layer_index = int(record["layer_index"])
        prefix = str(record["prefix"])
        plugin_name = str(record["plugin_name"])
        plugin = require_node(nodes_by_name, plugin_name, PLUGIN_OP)
        if plugin.domain != PLUGIN_DOMAIN or list(plugin.input) != [
            str(record["activation"]),
            str(record["packed_weight_name"]),
            str(record["packed_bias_name"]),
        ] or list(plugin.output) != [str(record["output"])]:
            raise RuntimeError("{} Plugin接口错误".format(plugin_name))

        attributes = node_attributes(plugin)
        expected_attributes = {
            "plugin_namespace": PLUGIN_NAMESPACE.encode("utf-8"),
            "plugin_version": PLUGIN_VERSION.encode("utf-8"),
            "stage": stage,
            "layer_index": layer_index,
            "input_height": int(record["height"]),
            "input_width": int(record["width"]),
            "input_channels": int(record["channels"]),
            "output_height": 11,
            "output_width": 11,
            "output_channels": int(record["channels"]),
            "kernel_height": int(record["sr_ratio"]),
            "kernel_width": int(record["sr_ratio"]),
            "stride_height": int(record["sr_ratio"]),
            "stride_width": int(record["sr_ratio"]),
            "groups": 1,
            "packed_k": int(record["packed_weight_shape"][0]),
            "packed_n": int(record["packed_weight_shape"][1]),
            "input_layout": b"BNC_FP16",
            "output_layout": b"BNC_FP16",
            "packed_weight_layout": PACKED_WEIGHT_LAYOUT.encode("utf-8"),
            "fuse_layernorm": 0,
        }
        for key, expected_value in expected_attributes.items():
            if attributes.get(key) != expected_value:
                raise RuntimeError(
                    "{} 属性{}错误: expected={}, actual={}".format(
                        plugin_name,
                        key,
                        expected_value,
                        attributes.get(key),
                    )
                )

        packed_weight = v17.initializer_array(
            initializers,
            str(record["packed_weight_name"]),
        )
        packed_bias = v17.initializer_array(
            initializers,
            str(record["packed_bias_name"]),
        )
        if packed_weight.dtype != np.float16 or tuple(packed_weight.shape) != tuple(record["packed_weight_shape"]):
            raise RuntimeError("{} packed weight dtype/shape错误".format(plugin_name))
        if packed_bias.dtype != np.float16 or tuple(packed_bias.shape) != tuple(record["packed_bias_shape"]):
            raise RuntimeError("{} packed bias dtype/shape错误".format(plugin_name))
        if array_sha256(packed_weight) != record["packed_weight_sha256"]:
            raise RuntimeError("{} packed weight内容变化".format(plugin_name))
        if array_sha256(packed_bias) != record["packed_bias_sha256"]:
            raise RuntimeError("{} packed bias内容变化".format(plugin_name))

        if str(record["old_weight_name"]) in initializers or str(record["old_bias_name"]) in initializers:
            raise RuntimeError("{} 原始 OIHW weight/bias仍残留".format(plugin_name))
        require_only_consumers(
            str(record["packed_weight_name"]),
            consumers,
            {plugin_name},
        )
        require_only_consumers(
            str(record["packed_bias_name"]),
            consumers,
            {plugin_name},
        )
        require_only_consumers(
            str(record["output"]),
            consumers,
            {
                prefix + "/norm/ReduceMean",
                prefix + "/norm/Sub",
            },
        )

        for old_name in record["remove_node_names"]:
            if old_name in nodes_by_name:
                raise RuntimeError("旧 SR 节点仍残留: {}".format(old_name))
        stage_counts[stage] += 1

    if stage_counts != {1: 3, 2: 4, 3: 18, 4: 0}:
        raise RuntimeError("SR Plugin stage数量错误: {}".format(stage_counts))
    block4_sr_plugins = [
        node.name
        for node in model.graph.node
        if node.op_type == PLUGIN_OP and node.name.startswith("/model/block4.")
    ]
    if block4_sr_plugins:
        raise RuntimeError("block4 不应存在 SR Plugin: {}".format(block4_sr_plugins))

    # 非目标节点必须逐字节保持 V22 状态，包含 Q/KV、LayerNorm、MHA 和旧 Plugin。
    for node_name, expected_hash in preserved_node_snapshot.items():
        node = nodes_by_name.get(node_name)
        if node is None:
            raise RuntimeError("非目标节点被删除: {}".format(node_name))
        if node_sha256(node) != expected_hash:
            raise RuntimeError("非目标节点发生变化: {}".format(node_name))

    after_plugin_snapshot = v17.plugin_snapshot(model)
    for plugin_name, expected_hash in original_plugin_snapshot.items():
        if after_plugin_snapshot.get(plugin_name) != expected_hash:
            raise RuntimeError("原有 Plugin发生变化: {}".format(plugin_name))

    # V24 只换 SR，不增加或删除 V22 的显式 Q/DQ。
    after_qdq = v17.qdq_count(model)
    if after_qdq["QuantizeLinear"] - before_qdq["QuantizeLinear"] != 101:
        raise RuntimeError("QuantizeLinear 数量不再等于 V22")
    if after_qdq["DequantizeLinear"] - before_qdq["DequantizeLinear"] != 101:
        raise RuntimeError("DequantizeLinear 数量不再等于 V22")

    # 再次确认 V22 的52个 Q/KV 投影边界完整。
    for _, _, _, target_name in v22.TARGET_SPECS:
        target = v17.require_node(nodes_by_name, target_name, "MatMul")
        v20.validate_qdq_pair(target, producer, initializers)

    print("\n[VALIDATE V24]")
    print("  block1 SR Plugin               : 3/3")
    print("  block2 SR Plugin               : 4/4")
    print("  block3 SR Plugin               : 18/18")
    print("  block4 SR Plugin               : 0 (no SR in model)")
    print("  total fused SR chains          : 25/25")
    print("  packed FP16 KxN weights        : 25/25")
    print("  V22 Q/KV projections preserved : 52/52")
    print("  original egcinet Plugins        : {}/{} unchanged".format(
        len(original_plugin_snapshot),
        len(original_plugin_snapshot),
    ))
    print("  total egcinet Plugins           : {}".format(len(after_plugin_snapshot)))


def rewrite_model(
    model: ModelProto,
    cache_scales: Mapping[str, float],
) -> Tuple[
    List[Dict[str, object]],
    Mapping[str, int],
    Dict[str, str],
    Dict[str, str],
]:
    """先生成 V22，再替换 block1～3 全部25条 SR 链。"""
    onnx.checker.check_model(model)

    # 步骤1：复用 V22 的完整重写和验证，确保当前性能基线一致。
    before_qdq, _ = v22.rewrite_model(model, cache_scales)
    ensure_plugin_opset(model)
    nodes_by_name, _, consumers, initializers = v17.build_maps(model)
    original_plugin_snapshot = v17.plugin_snapshot(model)
    node_order = {node.name: index for index, node in enumerate(model.graph.node)}

    # 步骤2：一次性验证25条链并准备 Plugin、packed weight/bias。
    records: List[Dict[str, object]] = []
    for stage, (depth, _, _, _, _) in STAGE_SPECS.items():
        for layer_index in range(depth):
            records.append(
                prepare_replacement(
                    model,
                    stage,
                    layer_index,
                    nodes_by_name,
                    consumers,
                    initializers,
                    node_order,
                )
            )

    remove_node_names: Set[str] = set()
    remove_tensor_names: Set[str] = set()
    remove_initializer_names: Set[str] = set()
    insertions: Dict[int, NodeProto] = {}
    new_initializers: List[TensorProto] = []
    for record in records:
        overlap = remove_node_names & set(record["remove_node_names"])
        if overlap:
            raise RuntimeError("SR 链节点重复: {}".format(sorted(overlap)))
        remove_node_names.update(record["remove_node_names"])
        remove_tensor_names.update(record["remove_tensor_names"])
        remove_initializer_names.update(
            [str(record["old_weight_name"]), str(record["old_bias_name"])]
        )
        insert_at = int(record["insert_at"])
        if insert_at in insertions:
            raise RuntimeError("Plugin 插入位置重复: {}".format(insert_at))
        insertions[insert_at] = record["plugin"]
        new_initializers.extend(
            [
                record["packed_weight_initializer"],
                record["packed_bias_initializer"],
            ]
        )

    # 记录所有非目标节点，保存/重载后也必须保持完全一致。
    preserved_node_snapshot = {
        node.name: node_sha256(node)
        for node in model.graph.node
        if node.name not in remove_node_names
    }

    # 步骤3：在原 Transpose_1 位置插入 Plugin，删除布局、Conv和专用 shape Constant。
    rewritten_nodes: List[NodeProto] = []
    for index, node in enumerate(model.graph.node):
        plugin = insertions.get(index)
        if plugin is not None:
            rewritten_nodes.append(plugin)
        if node.name not in remove_node_names:
            rewritten_nodes.append(node)
    if len(insertions) != len(records):
        raise RuntimeError("SR Plugin 没有全部插入")
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    # 步骤4：用 packed initializer 替换原始 OIHW weight/bias。
    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in remove_initializer_names
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    model.graph.initializer.extend(new_initializers)

    # 清理已删除中间 tensor 的 value_info；Plugin 最终输出名被复用，不在集合内。
    kept_value_info = [
        value_info
        for value_info in model.graph.value_info
        if value_info.name not in remove_tensor_names
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

    for record in records:
        print(
            "[REPLACE] block{}.{}, input=[1,{},{}], sr={}, "
            "packed_weight={}, output=[1,121,{}]".format(
                record["stage"],
                record["layer_index"],
                int(record["height"]) * int(record["width"]),
                record["channels"],
                record["sr_ratio"],
                record["packed_weight_shape"],
                record["channels"],
            )
        )

    validate_result(
        model,
        records,
        before_qdq,
        original_plugin_snapshot,
        preserved_node_snapshot,
    )
    return (
        records,
        before_qdq,
        original_plugin_snapshot,
        preserved_node_snapshot,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V24：V22基线上将block1～3全部25条SR布局+Conv链替换为packed-weight Plugin"
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v16_"
            "block4_packed_dwconv_gelu.onnx"
        ),
        help="稳定 V16 ONNX；脚本会先在内存中生成 V22",
    )
    parser.add_argument(
        "--cache",
        default="deploy/egcienet_352_multiclass_int8.cache",
        help="TensorRT INT8 calibration cache，用于重建 V22 Q/KV 边界",
    )
    parser.add_argument(
        "--output",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v24_"
            "q_block134_kv_all_packed_sr_plugin.onnx"
        ),
        help="25个 SR Plugin 节点的输出 ONNX",
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
    print("[SR    ] block1=3, block2=4, block3=18, block4=0; total=25")
    print("[PLUGIN] {} version {}".format(PLUGIN_OP, PLUGIN_VERSION))
    print("[PACK  ] {}".format(PACKED_WEIGHT_LAYOUT))

    cache_scales = v17.load_cache_scales(cache_path)
    model = onnx.load(str(input_path))
    (
        records,
        before_qdq,
        original_plugin_snapshot,
        preserved_node_snapshot,
    ) = rewrite_model(model, cache_scales)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)

    saved_model = onnx.load(str(output_path))
    validate_result(
        saved_model,
        records,
        before_qdq,
        original_plugin_snapshot,
        preserved_node_snapshot,
    )

    print("\n[DONE] {}".format(output_path))
    print(
        "[NEXT] 实现 Creator: name={}, version={}, namespace='{}'; "
        "三个输入依次为BNC激活、KxN packed FP16权重、FP16 bias".format(
            PLUGIN_OP,
            PLUGIN_VERSION,
            PLUGIN_NAMESPACE,
        )
    )


if __name__ == "__main__":
    main()
