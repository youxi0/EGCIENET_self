# EGCIENet: In-service blade defect detection

This version adds a TensorRT-friendly edge branch that distills offline SAM edge maps.
During training, SAM edge maps are used only as supervision for the edge branch.
During inference, SAM is not required.

The current edge branch is v2. It predicts a 1/4-scale edge map and directly
generates the first-stage edge guidance tokens, avoiding the expensive
full-resolution U-Net-like edge decoder used in the previous version. Checkpoints
from the previous edge branch are not architecture-compatible; retrain the model
after this change.

## Data layout

```text
Dataset/AEBIS/
  Train/
    JPEGImages/   # RGB images, e.g. 1.jpg
    BlackWhite/   # defect masks, e.g. 1.png
    Edge/         # offline SAM edge teachers, e.g. 1.jpg or 1.png
  Test/
    JPEGImages/
    BlackWhite/
    Edge/         # optional for inference
```

For multiclass semantic segmentation, convert the Labelme annotations in
`Dataset/AEBIS_Class` first:

```bash
python tools/convert_labelme_to_multiclass.py \
  --labelme-root ./Dataset/AEBIS_Class \
  --binary-root ./Dataset/AEBIS \
  --out-root ./Dataset/AEBIS_MultiClass
```

The converter writes:

```text
Dataset/AEBIS_MultiClass/
  classes.json
  Train/
    JPEGImages/
    SegClass/     # uint8 class-id masks, 0=background
    BlackWhite/   # binary projection, useful for sanity checks
    Edge/         # SAM edge teacher copied from AEBIS
  Test/
    JPEGImages/
    SegClass/
    BlackWhite/
    Edge/
```

The merged class IDs are similarity-based, not just sample-count based:

```text
0 background
1 burn            <- Burn
2 crack_tear      <- Crack, Tears
3 material_loss   <- Material missing, Nick
4 deformation     <- Dent, Tip curl
```

In the current local copy, `AEBIS` and `AEBIS_Class` mostly match by numeric
file stem, but not perfectly. `AEBIS_Class` is missing 10 Train annotations
(`248, 251, 254, 263, 266, 269, 274, 284, 501, 517`) and 1 Test annotation
(`260`). The converter skips these samples and writes split files containing
only the matched images. Samples without an edge teacher are also skipped by
default. If you explicitly want to create fallback edges from the segmentation
mask, add `--generate-missing-edge`.

YOLO segmentation datasets can also be converted to the same test layout. For
example, converting only the `test` split from an `images/{train,val,test}` and
`labels/{train,val,test}` YOLO directory:

```bash
python tools/convert_yolo_seg_to_egcienet.py \
  --images-root /path/to/aebad_yolo/images \
  --out-root ./Dataset/AEBAD_YOLO \
  --splits test
```

The converter writes `Test/JPEGImages`, `Test/BlackWhite`, `Test/SegClass`, and
`Test/test.txt`. By default, YOLO class ids are shifted by 1, so YOLO `0,1,2,3`
become mask ids `1,2,3,4`, while mask id `0` stays background. If the YOLO class
semantics do not match the AEBIS merged classes, use the binary metrics or pass
an explicit `--class-map`.

## Pretrained backbone

Training uses the MiT-B3 pretrained weights by default:

```text
mit_b3.pth
```

The original link:

```text
https://pan.baidu.com/s/11qnvFAbceMi4zuDI5YSYAA
Access code: tmx2
```

Place `mit_b3.pth` in the project root, or pass `--pretrained path/to/mit_b3.pth`.
If you only want to check whether the training pipeline can run, pass `--pretrained none`
to train from scratch.

## Dependencies

Besides PyTorch with CUDA, the code also needs:

```bash
pip install timm opencv-python
```

## Train

Recommended command for a 24 GB RTX 3090:

```bash
python train.py --train-root ./Dataset/AEBIS/Train/ --batch-size 16 --edge-loss-weight 1.0 --gpu 0
```

Multiclass semantic segmentation:

```bash
python train.py \
  --task multiclass \
  --train-root ./Dataset/AEBIS_MultiClass/Train/ \
  --class-config ./Dataset/AEBIS_MultiClass/classes.json \
  --batch-size 16 \
  --edge-loss-weight 1.0 \
  --gpu 0 \
  --amp
```

Optional mixed precision:

```bash
python train.py --train-root ./Dataset/AEBIS/Train/ --batch-size 16 --amp --gpu 0
```

Run without MiT-B3 pretrained weights:

```bash
python train.py --train-root ./Dataset/AEBIS/Train/ --batch-size 16 --edge-loss-weight 1.0 --gpu 0 --amp --pretrained none
```

The total training loss is:

```text
L = L_seg + edge_loss_weight * BCE(edge_pred, edge_sam)
```

`edge_sam` is downsampled to the edge branch output size before BCE. `L_seg`
keeps the original three-output deep supervision. Binary training uses BCE +
IoU. Multiclass training uses CrossEntropy + foreground Dice, so background
pixels do not dominate the Dice term.

## Test

```bash
python test.py --data-root ./Dataset/AEBIS/Test/ --model-path ./model/final.pth --out-path output/aebis/
```

Multiclass test:

```bash
python test.py \
  --task multiclass \
  --data-root ./Dataset/AEBIS_MultiClass/Test/ \
  --model-path ./model/final.pth \
  --out-path output/aebis_multiclass/ \
  --class-config ./Dataset/AEBIS_MultiClass/classes.json \
  --metrics-csv output/aebis_multiclass/metrics.csv \
  --gpu 0
```

For multiclass output, `out-path` stores grayscale class-id masks. The sibling
directory `out-path_color` stores color visualizations for inspection. The test
script reports both true multiclass metrics (`mIoU_fg`, per-class IoU/Dice) and
a foreground-vs-background projection for comparison with binary segmentation
results.

If `Dataset/AEBIS/Test/BlackWhite/` exists, the script will also print binary
segmentation metrics:

```text
Accuracy, Precision, Recall, Specificity, Dice/F1, IoU_fg, IoU_bg, mIoU, MAE
```

`IoU_fg` and `Dice/F1` are usually the most important values for defect
segmentation, because the defect pixels are the foreground class. Use
`--threshold` to change the binary mask threshold.

Save predicted edge maps for inspection:

```bash
python test.py --data-root ./Dataset/AEBIS/Test/ --model-path ./model/final.pth --out-path output/aebis/ --save-edge
```

Class-wise metrics are enabled automatically when `Dataset/AEBIS_Class.zip`
exists. This zip contains Labelme JSON files with defect labels, so the test
script groups test images by defect type and reports per-defect binary
segmentation quality:

```bash
python test.py \
  --data-root ./Dataset/AEBIS/Test/ \
  --model-path ./model/final.pth \
  --out-path output/aebis/ \
  --class-json-path ./Dataset/AEBIS_Class.zip \
  --metrics-csv output/aebis/metrics.csv
```

The known raw defect labels are:

```text
Burn, Dent, Material missing, Crack, Tears, Tip curl, Nick
```

Because the model predicts a binary defect mask rather than a defect category,
the class-wise table is computed by grouping images according to their Labelme
defect labels, then evaluating the binary mask within each group.

For deployment profiling, `pipeline speed` includes loading, saving, and metric
computation, while `model forward speed` measures only the PyTorch forward pass.

## Deployment

ONNX export and TensorRT FP16/INT8 engine build commands are documented in
[`DEPLOY.md`](DEPLOY.md).
