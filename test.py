import argparse
import csv
import json
import os
import time
import zipfile
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline import Mnet
from lib.class_config import class_names as config_class_names
from lib.class_config import load_class_config
from lib.class_config import num_classes as config_num_classes
from lib.dataset import Data


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
BINARY_METRIC_KEYS = (
    'Accuracy',
    'Precision',
    'Recall',
    'Specificity',
    'Dice/F1',
    'IoU_fg',
    'IoU_bg',
    'mIoU',
    'MAE',
)
CLASS_PALETTE_BGR = np.asarray(
    [
        [0, 0, 0],        # background
        [0, 0, 255],      # burn
        [0, 255, 255],    # crack_tear
        [0, 180, 0],      # material_loss
        [255, 0, 255],    # deformation
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Test EGCIENet with internal edge prediction.')
    parser.add_argument('--task', choices=('binary', 'multiclass'), default='binary')
    parser.add_argument('--data-root', default='./Dataset/AEBIS/Test/', help='Test dataset root.')
    parser.add_argument('--model-path', default='./model/final.pth', help='Checkpoint path.')
    parser.add_argument('--out-path', default='output/aebis/', help='Directory for predicted masks.')
    parser.add_argument('--class-config', default='./Dataset/AEBIS_MultiClass/classes.json')
    parser.add_argument('--num-classes', type=int, default=0, help='Override number of classes for multiclass.')
    parser.add_argument('--image-size', type=int, default=352)
    parser.add_argument('--edge-channels', type=int, default=16)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save-edge', action='store_true', help='Save predicted edge maps for inspection.')
    parser.add_argument('--no-metrics', action='store_true', help='Only save predictions, do not evaluate masks.')
    parser.add_argument(
        '--class-json-path',
        default='./Dataset/AEBIS_Class.zip',
        help='Optional Labelme JSON zip/directory for binary per-defect metrics. Use empty string to disable.',
    )
    parser.add_argument(
        '--class-map',
        default='',
        help='Optional JSON file mapping raw Labelme labels to merged class names for binary metrics.',
    )
    parser.add_argument('--metrics-csv', default='', help='Optional CSV path for metrics.')
    return parser.parse_args()


def to_int(value):
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def find_mask_path(mask_dir, stem):
    for ext in IMAGE_EXTS:
        path = os.path.join(mask_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def read_labelme_labels(raw_bytes):
    data = json.loads(raw_bytes.decode('utf-8'))
    labels = []
    for shape in data.get('shapes', []):
        label = str(shape.get('label', '')).strip()
        if label:
            labels.append(label)
    return sorted(set(labels))


def load_class_merge_map(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return {str(k).strip(): str(v).strip() for k, v in mapping.items()}


def apply_class_merge(labels, merge_map):
    if not merge_map:
        return labels
    merged = []
    for label in labels:
        class_name = merge_map.get(label, label)
        if class_name:
            merged.append(class_name)
    return sorted(set(merged))


def load_class_labels(path, merge_map=None):
    if not path or not os.path.exists(path):
        return {}

    labels_by_stem = {}
    if os.path.isfile(path):
        if not zipfile.is_zipfile(path):
            raise ValueError('class-json-path must be a zip file or directory: {}'.format(path))
        with zipfile.ZipFile(path, 'r') as zf:
            for name in zf.namelist():
                if not name.lower().endswith('.json'):
                    continue
                stem = os.path.splitext(os.path.basename(name))[0]
                labels = read_labelme_labels(zf.read(name))
                labels_by_stem[stem] = apply_class_merge(labels, merge_map or {})
    else:
        for root, _, files in os.walk(path):
            for file_name in files:
                if not file_name.lower().endswith('.json'):
                    continue
                stem = os.path.splitext(file_name)[0]
                json_path = os.path.join(root, file_name)
                with open(json_path, 'rb') as f:
                    labels = read_labelme_labels(f.read())
                labels_by_stem[stem] = apply_class_merge(labels, merge_map or {})

    return labels_by_stem


def print_binary_metrics(name, result):
    print('{} on {} images:'.format(name, result['images']))
    for key in BINARY_METRIC_KEYS:
        print('{}: {:.6f}'.format(key, result[key]))


def print_binary_class_metrics(rows):
    if not rows:
        return
    print('per-defect metrics:')
    header = '{:<20} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
        'Class', 'Images', 'IoU_fg', 'Dice/F1', 'Precision', 'Recall', 'MAE'
    )
    print(header)
    print('-' * len(header))
    for class_name, result in rows:
        print(
            '{:<20} {:>6d} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}'.format(
                class_name[:20],
                int(result['images']),
                result['IoU_fg'],
                result['Dice/F1'],
                result['Precision'],
                result['Recall'],
                result['MAE'],
            )
        )


def save_binary_metrics_csv(path, global_result, class_rows):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = ['split', 'images'] + list(BINARY_METRIC_KEYS)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row = {'split': 'all', 'images': global_result['images']}
        for key in BINARY_METRIC_KEYS:
            row[key] = '{:.6f}'.format(global_result[key])
        writer.writerow(row)

        for class_name, result in class_rows:
            row = {'split': class_name, 'images': result['images']}
            for key in BINARY_METRIC_KEYS:
                row[key] = '{:.6f}'.format(result[key])
            writer.writerow(row)
    print('metrics saved to {}'.format(path))


class BinarySegMetrics(object):
    def __init__(self, eps=1e-7):
        self.eps = eps
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.mae_sum = 0.0
        self.pixel_count = 0
        self.image_count = 0

    def update(self, pred_prob, pred_bin, gt_bin):
        pred_bin = pred_bin.astype(bool)
        gt_bin = gt_bin.astype(bool)

        self.tp += int(np.logical_and(pred_bin, gt_bin).sum())
        self.fp += int(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
        self.fn += int(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())
        self.tn += int(np.logical_and(np.logical_not(pred_bin), np.logical_not(gt_bin)).sum())
        self.mae_sum += float(np.abs(pred_prob.astype(np.float32) - gt_bin.astype(np.float32)).sum())
        self.pixel_count += int(gt_bin.size)
        self.image_count += 1

    def compute(self):
        tp = float(self.tp)
        fp = float(self.fp)
        fn = float(self.fn)
        tn = float(self.tn)
        eps = self.eps

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        iou_fg = tp / (tp + fp + fn + eps)
        iou_bg = tn / (tn + fp + fn + eps)
        miou = 0.5 * (iou_fg + iou_bg)
        mae = self.mae_sum / max(self.pixel_count, 1)

        return {
            'images': self.image_count,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'Dice/F1': dice,
            'IoU_fg': iou_fg,
            'IoU_bg': iou_bg,
            'mIoU': miou,
            'MAE': mae,
        }


class MultiClassSegMetrics(object):
    def __init__(self, num_classes, ignore_index=255, eps=1e-7):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.class_image_count = np.zeros((num_classes,), dtype=np.int64)
        self.image_count = 0

    def update(self, pred_cls, gt_cls):
        pred_cls = pred_cls.astype(np.int64)
        gt_cls = gt_cls.astype(np.int64)
        valid = (
            (gt_cls != self.ignore_index)
            & (gt_cls >= 0)
            & (gt_cls < self.num_classes)
            & (pred_cls >= 0)
            & (pred_cls < self.num_classes)
        )
        if not np.any(valid):
            return

        self.image_count += 1
        gt_valid = gt_cls[valid]
        pred_valid = pred_cls[valid]
        for class_id in np.unique(gt_valid):
            self.class_image_count[int(class_id)] += 1

        hist = np.bincount(
            self.num_classes * gt_valid + pred_valid,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        self.hist += hist

    def compute(self):
        hist = self.hist.astype(np.float64)
        tp = np.diag(hist)
        gt_sum = hist.sum(axis=1)
        pred_sum = hist.sum(axis=0)
        union = gt_sum + pred_sum - tp

        iou = np.full(self.num_classes, np.nan, dtype=np.float64)
        dice = np.full(self.num_classes, np.nan, dtype=np.float64)
        precision = np.full(self.num_classes, np.nan, dtype=np.float64)
        recall = np.full(self.num_classes, np.nan, dtype=np.float64)

        union_mask = union > 0
        dice_mask = (gt_sum + pred_sum) > 0
        pred_mask = pred_sum > 0
        gt_mask = gt_sum > 0

        iou[union_mask] = tp[union_mask] / (union[union_mask] + self.eps)
        dice[dice_mask] = 2.0 * tp[dice_mask] / (gt_sum[dice_mask] + pred_sum[dice_mask] + self.eps)
        precision[pred_mask] = tp[pred_mask] / (pred_sum[pred_mask] + self.eps)
        recall[gt_mask] = tp[gt_mask] / (gt_sum[gt_mask] + self.eps)

        fg_mask = gt_mask.copy()
        fg_mask[0] = False
        present_mask = gt_mask

        pixel_acc = tp.sum() / max(hist.sum(), self.eps)
        miou_all = float(np.nanmean(iou[present_mask])) if np.any(present_mask) else 0.0
        miou_fg = float(np.nanmean(iou[fg_mask])) if np.any(fg_mask) else 0.0
        mdice_fg = float(np.nanmean(dice[fg_mask])) if np.any(fg_mask) else 0.0

        return {
            'images': self.image_count,
            'pixel_accuracy': float(pixel_acc),
            'mIoU_all': miou_all,
            'mIoU_fg': miou_fg,
            'mDice_fg': mdice_fg,
            'class_images': self.class_image_count.copy(),
            'class_pixels': gt_sum.astype(np.int64),
            'IoU': iou,
            'Dice/F1': dice,
            'Precision': precision,
            'Recall': recall,
        }


def metric_value(value):
    if np.isnan(value):
        return 'nan'
    return '{:.6f}'.format(float(value))


def print_multiclass_metrics(result, names):
    print('multiclass metrics on {} images:'.format(result['images']))
    print('Pixel Accuracy: {:.6f}'.format(result['pixel_accuracy']))
    print('mIoU_all: {:.6f}'.format(result['mIoU_all']))
    print('mIoU_fg: {:.6f}'.format(result['mIoU_fg']))
    print('mDice_fg: {:.6f}'.format(result['mDice_fg']))
    print('per-class metrics:')
    header = '{:<4} {:<20} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
        'ID', 'Class', 'Images', 'Pixels', 'IoU', 'Dice/F1', 'Precision', 'Recall'
    )
    print(header)
    print('-' * len(header))
    for class_id in range(len(result['IoU'])):
        print(
            '{:<4d} {:<20} {:>6d} {:>10d} {:>10} {:>10} {:>10} {:>10}'.format(
                class_id,
                names[class_id][:20],
                int(result['class_images'][class_id]),
                int(result['class_pixels'][class_id]),
                metric_value(result['IoU'][class_id]),
                metric_value(result['Dice/F1'][class_id]),
                metric_value(result['Precision'][class_id]),
                metric_value(result['Recall'][class_id]),
            )
        )


def save_multiclass_metrics_csv(path, result, names, binary_result=None):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        'scope',
        'class_id',
        'class_name',
        'images',
        'pixels',
        'Pixel Accuracy',
        'mIoU_all',
        'mIoU_fg',
        'mDice_fg',
        'IoU',
        'Dice/F1',
        'Precision',
        'Recall',
        'Specificity',
        'IoU_bg',
        'MAE',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                'scope': 'multiclass_overall',
                'class_id': '',
                'class_name': '',
                'images': result['images'],
                'pixels': int(result['class_pixels'].sum()),
                'Pixel Accuracy': '{:.6f}'.format(result['pixel_accuracy']),
                'mIoU_all': '{:.6f}'.format(result['mIoU_all']),
                'mIoU_fg': '{:.6f}'.format(result['mIoU_fg']),
                'mDice_fg': '{:.6f}'.format(result['mDice_fg']),
            }
        )
        if binary_result is not None:
            writer.writerow(
                {
                    'scope': 'foreground_binary',
                    'class_id': '',
                    'class_name': 'foreground',
                    'images': binary_result['images'],
                    'pixels': '',
                    'Pixel Accuracy': '{:.6f}'.format(binary_result['Accuracy']),
                    'mIoU_fg': '{:.6f}'.format(binary_result['IoU_fg']),
                    'IoU': '{:.6f}'.format(binary_result['IoU_fg']),
                    'Dice/F1': '{:.6f}'.format(binary_result['Dice/F1']),
                    'Precision': '{:.6f}'.format(binary_result['Precision']),
                    'Recall': '{:.6f}'.format(binary_result['Recall']),
                    'Specificity': '{:.6f}'.format(binary_result['Specificity']),
                    'IoU_bg': '{:.6f}'.format(binary_result['IoU_bg']),
                    'MAE': '{:.6f}'.format(binary_result['MAE']),
                }
            )
        for class_id in range(len(result['IoU'])):
            writer.writerow(
                {
                    'scope': 'class',
                    'class_id': class_id,
                    'class_name': names[class_id],
                    'images': int(result['class_images'][class_id]),
                    'pixels': int(result['class_pixels'][class_id]),
                    'IoU': metric_value(result['IoU'][class_id]),
                    'Dice/F1': metric_value(result['Dice/F1'][class_id]),
                    'Precision': metric_value(result['Precision'][class_id]),
                    'Recall': metric_value(result['Recall'][class_id]),
                }
            )
    print('metrics saved to {}'.format(path))


def colorize_class_mask(mask, num_classes):
    if num_classes <= len(CLASS_PALETTE_BGR):
        palette = CLASS_PALETTE_BGR
    else:
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        palette[:len(CLASS_PALETTE_BGR)] = CLASS_PALETTE_BGR
        for class_id in range(len(CLASS_PALETTE_BGR), num_classes):
            palette[class_id] = [
                (37 * class_id) % 255,
                (97 * class_id) % 255,
                (173 * class_id) % 255,
            ]
    clipped = np.clip(mask, 0, len(palette) - 1)
    return palette[clipped]


def resolve_multiclass_meta(args, checkpoint):
    config = load_class_config(args.class_config)
    checkpoint_classes = 0
    if isinstance(checkpoint, dict):
        checkpoint_classes = int(checkpoint.get('num_classes', 0) or 0)

    if args.num_classes > 0:
        num_classes = args.num_classes
    elif checkpoint_classes > 1:
        num_classes = checkpoint_classes
    else:
        num_classes = config_num_classes(config)

    names = config_class_names(config)
    while len(names) < num_classes:
        names.append('class_{}'.format(len(names)))
    return num_classes, names[:num_classes]


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.out_path, exist_ok=True)

    edge_out_path = args.out_path.rstrip('/\\') + '_edge'
    if args.save_edge:
        os.makedirs(edge_out_path, exist_ok=True)

    color_out_path = args.out_path.rstrip('/\\') + '_color'
    if args.task == 'multiclass':
        os.makedirs(color_out_path, exist_ok=True)

    gt_dir = os.path.join(args.data_root, 'SegClass' if args.task == 'multiclass' else 'BlackWhite')
    eval_metrics = (not args.no_metrics) and os.path.isdir(gt_dir)

    print('loading model from {}...'.format(args.model_path))
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    if args.task == 'multiclass':
        num_classes, names = resolve_multiclass_meta(args, checkpoint)
    else:
        num_classes, names = 1, ['background', 'defect']

    data = Data(root=args.data_root, mode='test', image_size=args.image_size, task=args.task)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    if args.task == 'multiclass':
        metrics = MultiClassSegMetrics(num_classes) if eval_metrics else None
        binary_metrics = BinarySegMetrics() if eval_metrics else None
        class_lookup = {}
        class_metrics = {}
        missing_class_labels = 0
    else:
        metrics = BinarySegMetrics() if eval_metrics else None
        class_lookup = {}
        class_metrics = defaultdict(BinarySegMetrics)
        missing_class_labels = 0
        if eval_metrics and args.class_json_path:
            merge_map = load_class_merge_map(args.class_map)
            class_lookup = load_class_labels(args.class_json_path, merge_map)
            if class_lookup:
                class_names = sorted({label for labels in class_lookup.values() for label in labels})
                print('loaded class labels for {} images: {}'.format(len(class_lookup), ', '.join(class_names)))
            else:
                print('class metrics skipped: class-json-path not found at {}'.format(args.class_json_path))

    net = Mnet(pretrained=False, edge_channels=args.edge_channels, num_classes=num_classes).cuda()
    net.load_state_dict(state_dict)
    net.eval()
    print('task: {}, num_classes: {}'.format(args.task, num_classes))

    img_num = len(loader)
    time_s = time.time()
    infer_time = 0.0
    with torch.no_grad():
        for rgb, (h, w), name in loader:
            h = to_int(h)
            w = to_int(w)
            rgb = rgb.cuda().float()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_s = time.time()
            score1, score2, score3, prob1, prob2, prob3, edge_logit, edge_prob = net(rgb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time += time.time() - infer_s

            score1 = F.interpolate(score1, size=(h, w), mode='bilinear', align_corners=True)
            mask_name = os.path.splitext(name[0])[0] + '.png'
            stem = os.path.splitext(name[0])[0]

            if args.task == 'multiclass':
                prob_tensor = F.softmax(score1, dim=1)
                pred_cls = torch.argmax(prob_tensor, dim=1)[0].cpu().numpy().astype(np.uint8)
                prob = prob_tensor[0].cpu().numpy()
                cv2.imwrite(os.path.join(args.out_path, mask_name), pred_cls)
                cv2.imwrite(os.path.join(color_out_path, mask_name), colorize_class_mask(pred_cls, num_classes))

                if eval_metrics:
                    gt_path = find_mask_path(gt_dir, stem)
                    if gt_path is not None:
                        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        if gt is None:
                            raise FileNotFoundError('Could not read mask: {}'.format(gt_path))
                        if gt.shape != pred_cls.shape:
                            gt = cv2.resize(
                                gt,
                                dsize=(pred_cls.shape[1], pred_cls.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        metrics.update(pred_cls, gt)
                        valid = gt != 255
                        if np.any(valid):
                            binary_metrics.update((1.0 - prob[0])[valid], (pred_cls > 0)[valid], (gt > 0)[valid])
            else:
                pred = np.squeeze(torch.sigmoid(score1).cpu().data.numpy())
                pred_bin = pred > args.threshold
                pred_save = pred_bin.astype(np.uint8) * 255
                cv2.imwrite(os.path.join(args.out_path, mask_name), pred_save)

                if eval_metrics:
                    gt_path = find_mask_path(gt_dir, stem)
                    if gt_path is not None:
                        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        if gt is None:
                            raise FileNotFoundError('Could not read mask: {}'.format(gt_path))
                        if gt.shape != pred.shape:
                            gt = cv2.resize(gt, dsize=(pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                        gt_bin = gt > 127
                        metrics.update(pred, pred_bin, gt_bin)

                        if class_lookup:
                            labels = class_lookup.get(stem, [])
                            if labels:
                                for class_name in labels:
                                    class_metrics[class_name].update(pred, pred_bin, gt_bin)
                            else:
                                missing_class_labels += 1

            if args.save_edge:
                edge = F.interpolate(edge_prob, size=(h, w), mode='bilinear', align_corners=True)
                edge = np.squeeze(edge.cpu().data.numpy())
                edge = np.clip(edge * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(edge_out_path, mask_name), edge)

            print('{} Done!'.format(name[0]))

    time_e = time.time()
    print('pipeline speed: {:.6f} FPS'.format(img_num / (time_e - time_s)))
    if infer_time > 0:
        print('model forward speed: {:.6f} FPS'.format(img_num / infer_time))

    if eval_metrics and metrics.image_count > 0:
        if args.task == 'multiclass':
            result = metrics.compute()
            binary_result = binary_metrics.compute()
            print_multiclass_metrics(result, names)
            print('foreground-vs-background projection:')
            print_binary_metrics('foreground binary metrics', binary_result)
            save_multiclass_metrics_csv(args.metrics_csv, result, names, binary_result)
        else:
            result = metrics.compute()
            print('threshold={:.2f}'.format(args.threshold))
            print_binary_metrics('overall metrics', result)
            class_rows = []
            for class_name in sorted(class_metrics.keys()):
                class_rows.append((class_name, class_metrics[class_name].compute()))
            print_binary_class_metrics(class_rows)
            if missing_class_labels:
                print('class labels missing for {} evaluated images'.format(missing_class_labels))
            save_binary_metrics_csv(args.metrics_csv, result, class_rows)
    elif not args.no_metrics:
        print('metrics skipped: ground-truth directory not found at {}'.format(gt_dir))
