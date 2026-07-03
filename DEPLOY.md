# EGCIENet deployment notes

This deployment path is intended for Jetson Orin Nano:

```text
PyTorch checkpoint -> ONNX FP32 -> TensorRT FP16 / INT8 engine
```

## 1. Model input and output

The exported deployment model has one input:

```text
name: image
shape: [1, 3, 352, 352]
dtype: float32
layout: NCHW
color: BGR, same as cv2.imread
```

Preprocess each image as:

```python
mean = [0.551, 0.619, 0.532] * 255
std  = [0.241, 0.236, 0.244] * 255
image = cv2.resize(image_bgr, (352, 352))
image = (image.astype(np.float32) - mean) / std
image = image.transpose(2, 0, 1)[None]
```

Default output:

```text
name: mask
shape: [1, 1, 352, 352]
dtype: float32
range: 0-1 probability
```

Use `mask > 0.5` as the binary defect mask, then resize it back to the
original image size if needed.

If `--output-edge` is used during export, the ONNX model has a second output:

```text
name: edge
shape: [1, 1, 352, 352]
dtype: float32
range: 0-1 probability
```

The edge output is for visualization/debugging. Production deployment usually
only needs `mask`.

## 2. Export ONNX

Run this on the training server or any machine with PyTorch installed:

```bash
python deploy/export_onnx.py \
  --model-path ./model/final.pth \
  --onnx-path ./deploy/egcienet_352.onnx \
  --image-size 352 \
  --batch-size 1 \
  --device cuda
```

Export with edge output:

```bash
python deploy/export_onnx.py \
  --model-path ./model/final.pth \
  --onnx-path ./deploy/egcienet_352_edge.onnx \
  --image-size 352 \
  --batch-size 1 \
  --device cuda \
  --output-edge
```

The script prints the PyTorch deploy I/O and ONNX graph I/O.

Optional dependencies for ONNX checking:

```bash
pip install onnx onnxruntime-gpu
```

## 3. Validate ONNX

Compare PyTorch and ONNX Runtime outputs before building TensorRT:

```bash
python deploy/validate_onnx.py \
  --model-path ./model/final.pth \
  --onnx-path ./deploy/egcienet_352.onnx \
  --image-path ./Dataset/AEBIS/Test/JPEGImages/0.jpg \
  --image-size 352 \
  --device cuda \
  --save-mask ./deploy/onnx_mask_0.png
```

Expected result: the max/mean absolute difference should be small. Small
numerical differences are normal.

## 4. Build TensorRT FP16 engine

Recommended on Jetson Orin Nano:

```bash
python deploy/build_tensorrt_engine.py \
  --onnx ./deploy/egcienet_352.onnx \
  --engine ./deploy/egcienet_352_fp16.engine \
  --precision fp16 \
  --image-size 352 \
  --batch-size 1 \
  --workspace 2048
```

Alternative with `trtexec`:

```bash
trtexec \
  --onnx=./deploy/egcienet_352.onnx \
  --saveEngine=./deploy/egcienet_352_fp16.engine \
  --fp16 \
  --shapes=image:1x3x352x352 \
  --memPoolSize=workspace:2048
```

For older TensorRT versions, replace `--memPoolSize=workspace:2048` with:

```bash
--workspace=2048
```

## 5. Build TensorRT INT8 engine

INT8 needs calibration images. Use training images and the same preprocessing as
the model:

```bash
python deploy/build_tensorrt_engine.py \
  --onnx ./deploy/egcienet_352.onnx \
  --engine ./deploy/egcienet_352_int8.engine \
  --precision int8 \
  --image-size 352 \
  --batch-size 1 \
  --workspace 2048 \
  --calib-root ./Dataset/AEBIS/Train/JPEGImages \
  --calib-cache ./deploy/int8_calib.cache \
  --num-calib 300
```

Notes:

- FP16 is the first deployment target because it usually keeps accuracy close to
  PyTorch and is fast on Orin Nano.
- INT8 should be accepted only after running the test set again and comparing
  `IoU_fg`, `Dice/F1`, `Recall`, and `MAE` with FP32/FP16.
- Use fixed batch 1 and fixed 352x352 input for the first Jetson deployment.
  Dynamic shape is possible, but fixed shape is simpler and usually faster.

## 6. Recommended deployment check order

```text
1. PyTorch test.py metrics
2. export_onnx.py
3. validate_onnx.py
4. TensorRT FP16 engine
5. FP16 test-set metrics and FPS
6. TensorRT INT8 engine with calibration
7. INT8 test-set metrics and FPS
```
