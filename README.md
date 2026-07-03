# EGCIENet: In-service blade defect detection

This version adds a lightweight edge branch that distills offline SAM edge maps.
During training, SAM edge maps are used only as supervision for the edge branch.
During inference, SAM is not required.

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

`L_seg` keeps the original three-output deep supervision with BCE + IoU.

## Test

```bash
python test.py --data-root ./Dataset/AEBIS/Test/ --model-path ./model/final.pth --out-path output/aebis/
```

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
