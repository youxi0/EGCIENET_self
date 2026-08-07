python - <<'PY'
import onnx
from collections import Counter

path = "deploy/egcienet_352_multiclass_qdq_v1.onnx"
model = onnx.load(path)
onnx.checker.check_model(model)

producer = {
    output: node
    for node in model.graph.node
    for output in node.output
}

expected = {
    f"/model/decoder/FEM{f}/conv{c}/Conv"
    for f in range(1, 5)
    for c in range(1, 4)
}
expected |= {
    f"/model/decoder/{name}/conv_pre/conv_pre.0/Conv"
    for name in ("d31", "d42", "d42_31")
}

qdq_count = Counter(
    node.op_type
    for node in model.graph.node
    if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
)

quantized_conv = set()
for node in model.graph.node:
    if node.op_type != "Conv":
        continue

    if any(
        input_name in producer
        and producer[input_name].op_type == "DequantizeLinear"
        for input_name in node.input
    ):
        quantized_conv.add(node.name)

print("Q/DQ:", dict(qdq_count))
print("Quantized Conv:", len(quantized_conv))

for name in sorted(quantized_conv):
    print(name)

missing = expected - quantized_conv
unexpected = quantized_conv - expected

print("\nMissing:", sorted(missing))
print("Unexpected:", sorted(unexpected))

assert not missing, f"目标卷积缺失: {sorted(missing)}"
assert not unexpected, f"出现非目标量化卷积: {sorted(unexpected)}"
assert len(quantized_conv) == 15

print("\n[PASS] 显式 Q/DQ 只覆盖预期的 15 个 Decoder Conv")
PY
