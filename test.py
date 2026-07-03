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
from lib.dataset import Data


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
METRIC_KEYS = ('Accuracy', 'Precision', 'Recall', 'Specificity', 'Dice/F1', 'IoU_fg', 'IoU_bg', 'mIoU', 'MAE')


def parse_args():
    parser = argparse.ArgumentParser(description='Test EGCIENet with internal edge prediction.')
    parser.add_argument('--data-root', default='./Dataset/AEBIS/Test/', help='Test dataset root.')
    parser.add_argument('--model-path', default='./model/final.pth', help='Checkpoint path.')
    parser.add_argument('--out-path', default='output/aebis/', help='Directory for predicted masks.')
    parser.add_argument('--image-size', type=int, default=352)
    parser.add_argument('--edge-channels', type=int, default=32)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save-edge', action='store_true', help='Save predicted edge maps for inspection.')
    parser.add_argument('--no-metrics', action='store_true', help='Only save predictions, do not evaluate masks.')
    parser.add_argument(
        '--class-json-path',
        default='./Dataset/AEBIS_Class.zip',
        help='Optional Labelme JSON zip/directory for per-defect metrics. Use empty string to disable.',
    )
    parser.add_argument(
        '--class-map',
        default='',
        help='Optional JSON file mapping raw Labelme labels to merged class names.',
    )
    parser.add_argument('--metrics-csv', default='', help='Optional CSV path for global and class-wise metrics.')
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


def print_metrics(name, result):
    print('{} on {} images:'.format(name, result['images']))
    for key in METRIC_KEYS:
        print('{}: {:.6f}'.format(key, result[key]))


def print_class_metrics(rows):
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


def save_metrics_csv(path, global_result, class_rows):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = ['split', 'images'] + list(METRIC_KEYS)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row = {'split': 'all', 'images': global_result['images']}
        for key in METRIC_KEYS:
            row[key] = '{:.6f}'.format(global_result[key])
        writer.writerow(row)

        for class_name, result in class_rows:
            row = {'split': class_name, 'images': result['images']}
            for key in METRIC_KEYS:
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


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.out_path, exist_ok=True)

    edge_out_path = args.out_path.rstrip('/\\') + '_edge'
    if args.save_edge:
        os.makedirs(edge_out_path, exist_ok=True)

    data = Data(root=args.data_root, mode='test', image_size=args.image_size)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    gt_dir = os.path.join(args.data_root, 'BlackWhite')
    eval_metrics = (not args.no_metrics) and os.path.isdir(gt_dir)
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

    net = Mnet(pretrained=False, edge_channels=args.edge_channels).cuda()
    print('loading model from {}...'.format(args.model_path))
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()

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
            score1, score2, score3, s1, s2, s3, edge_logit, edge_prob = net(rgb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time += time.time() - infer_s

            score1 = F.interpolate(score1, size=(h, w), mode='bilinear', align_corners=True)

            pred = np.squeeze(torch.sigmoid(score1).cpu().data.numpy())
            pred_bin = pred > args.threshold
            pred_save = pred_bin.astype(np.uint8) * 255
            mask_name = os.path.splitext(name[0])[0] + '.png'
            cv2.imwrite(os.path.join(args.out_path, mask_name), pred_save)

            if eval_metrics:
                stem = os.path.splitext(name[0])[0]
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
        result = metrics.compute()
        print('threshold={:.2f}'.format(args.threshold))
        print_metrics('overall metrics', result)
        class_rows = []
        for class_name in sorted(class_metrics.keys()):
            class_rows.append((class_name, class_metrics[class_name].compute()))
        print_class_metrics(class_rows)
        if missing_class_labels:
            print('class labels missing for {} evaluated images'.format(missing_class_labels))
        save_metrics_csv(args.metrics_csv, result, class_rows)
    elif not args.no_metrics:
        print('metrics skipped: ground-truth directory not found at {}'.format(gt_dir))
