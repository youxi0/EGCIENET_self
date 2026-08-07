import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
SPLIT_NAME = {
    'train': 'Train',
    'val': 'Val',
    'test': 'Test',
}


def parse_class_map(value, class_offset):
    if not value:
        return None

    mapping = {}
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        src, dst = item.split(':', 1)
        mapping[int(src)] = int(dst)
    return mapping


def mapped_class_id(yolo_class_id, class_map, class_offset):
    if class_map is None:
        return int(yolo_class_id) + int(class_offset)
    return int(class_map.get(int(yolo_class_id), -1))


def find_image(image_dir, stem):
    for ext in IMAGE_EXTS:
        path = image_dir / (stem + ext)
        if path.exists():
            return path
    return None


def iter_images(image_dir):
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def normalized_point(x, y, width, height):
    px = max(0, min(width - 1, int(round(float(x) * width))))
    py = max(0, min(height - 1, int(round(float(y) * height))))
    return px, py


def parse_yolo_line(line, width, height):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    yolo_class_id = int(float(parts[0]))
    values = [float(item) for item in parts[1:]]

    if len(values) == 4:
        cx, cy, bw, bh = values
        x1 = max(0, min(width - 1, int(round((cx - bw / 2.0) * width))))
        y1 = max(0, min(height - 1, int(round((cy - bh / 2.0) * height))))
        x2 = max(0, min(width - 1, int(round((cx + bw / 2.0) * width))))
        y2 = max(0, min(height - 1, int(round((cy + bh / 2.0) * height))))
        return yolo_class_id, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], 'box'

    if len(values) % 2 != 0:
        return None

    points = [
        normalized_point(values[i], values[i + 1], width, height)
        for i in range(0, len(values), 2)
    ]
    if len(points) < 3:
        return None
    return yolo_class_id, points, 'polygon'


def build_masks(label_path, width, height, class_map, class_offset):
    binary = Image.new('L', (width, height), 0)
    seg = Image.new('L', (width, height), 0)
    binary_draw = ImageDraw.Draw(binary)
    seg_draw = ImageDraw.Draw(seg)
    raw_counter = Counter()
    mapped_counter = Counter()
    invalid_lines = 0
    skipped_lines = 0
    polygon_lines = 0
    box_lines = 0

    if not label_path.exists():
        return binary, seg, raw_counter, mapped_counter, 0, 0, 0, 1

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_yolo_line(line, width, height)
            if parsed is None:
                invalid_lines += 1
                continue
            yolo_class_id, points, shape_type = parsed
            target_id = mapped_class_id(yolo_class_id, class_map, class_offset)
            if target_id < 0:
                skipped_lines += 1
                continue

            binary_draw.polygon(points, outline=255, fill=255)
            seg_draw.polygon(points, outline=target_id, fill=target_id)
            raw_counter[yolo_class_id] += 1
            mapped_counter[target_id] += 1
            if shape_type == 'polygon':
                polygon_lines += 1
            else:
                box_lines += 1

    return binary, seg, raw_counter, mapped_counter, polygon_lines, box_lines, invalid_lines, skipped_lines


def write_class_config(path, mapped_counter, class_names):
    max_class_id = max([0] + list(mapped_counter.keys()))
    classes = [{'id': 0, 'name': 'background', 'raw_labels': []}]

    reverse_raw = {}
    for class_id in mapped_counter.keys():
        reverse_raw.setdefault(int(class_id), [])

    for class_id in range(1, max_class_id + 1):
        name_idx = class_id - 1
        if name_idx < len(class_names):
            name = class_names[name_idx]
        else:
            name = 'yolo_class_{}'.format(class_id - 1)
        classes.append({'id': class_id, 'name': name, 'raw_labels': reverse_raw.get(class_id, [])})

    config = {
        'version': 1,
        'background_id': 0,
        'merge_strategy': 'yolo_segmentation_conversion',
        'classes': classes,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def prepare_split_dirs(out_root, split):
    split_root = out_root / SPLIT_NAME[split]
    dirs = {
        'images': split_root / 'JPEGImages',
        'binary': split_root / 'BlackWhite',
        'seg': split_root / 'SegClass',
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return split_root, dirs


def convert_split(images_root, labels_root, out_root, split, class_map, class_offset, copy_images):
    image_dir = images_root / split
    label_dir = labels_root / split
    if not image_dir.exists():
        print('split skipped, image dir not found: {}'.format(image_dir))
        return Counter(), Counter(), 0

    split_root, dirs = prepare_split_dirs(out_root, split)
    raw_total = Counter()
    mapped_total = Counter()
    written = 0
    missing_label = 0
    invalid_lines = 0
    skipped_lines = 0
    polygon_lines = 0
    box_lines = 0

    with open(split_root / (split.lower() + '.txt'), 'w', encoding='utf-8') as split_file:
        for image_path in iter_images(image_dir):
            label_path = label_dir / (image_path.stem + '.txt')
            with Image.open(image_path) as image:
                width, height = image.size

            binary, seg, raw_counter, mapped_counter, polygons, boxes, invalid, skipped = build_masks(
                label_path,
                width,
                height,
                class_map,
                class_offset,
            )
            if not label_path.exists():
                missing_label += 1

            if copy_images:
                shutil.copy2(image_path, dirs['images'] / image_path.name)
            else:
                with Image.open(image_path) as image:
                    image.convert('RGB').save(dirs['images'] / (image_path.stem + '.jpg'), quality=95)

            binary.save(dirs['binary'] / (image_path.stem + '.png'))
            seg.save(dirs['seg'] / (image_path.stem + '.png'))
            split_file.write(image_path.stem + image_path.suffix.lower() + '\n')

            raw_total.update(raw_counter)
            mapped_total.update(mapped_counter)
            polygon_lines += polygons
            box_lines += boxes
            invalid_lines += invalid
            skipped_lines += skipped
            written += 1

    print(
        '{}: written {}, missing_label {}, polygon_lines {}, box_lines {}, invalid_lines {}, skipped_lines {}'.format(
            split,
            written,
            missing_label,
            polygon_lines,
            box_lines,
            invalid_lines,
            skipped_lines,
        )
    )
    return raw_total, mapped_total, written


def parse_args():
    parser = argparse.ArgumentParser(description='Convert YOLO segmentation labels to EGCIENet dataset layout.')
    parser.add_argument('--images-root', required=True, help='YOLO images root, e.g. /path/images.')
    parser.add_argument('--labels-root', default='', help='YOLO labels root. Default: sibling labels directory.')
    parser.add_argument('--out-root', default='./Dataset/AEBAD_YOLO', help='Output dataset root.')
    parser.add_argument('--splits', nargs='+', default=['test'], choices=['train', 'val', 'test', 'all'])
    parser.add_argument(
        '--class-map',
        default='',
        help='Map YOLO class ids to output mask ids, e.g. 0:1,1:2,2:3,3:4. Unmapped ids are skipped.',
    )
    parser.add_argument(
        '--class-offset',
        type=int,
        default=1,
        help='Default mapping when --class-map is empty: output_id = yolo_id + class_offset.',
    )
    parser.add_argument(
        '--class-names',
        nargs='*',
        default=[],
        help='Optional output class names for ids 1..N.',
    )
    parser.add_argument(
        '--reencode-images',
        action='store_true',
        help='Re-encode images as RGB jpg instead of copying originals.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root) if args.labels_root else images_root.parent / 'labels'
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    class_map = parse_class_map(args.class_map, args.class_offset)
    splits = ['train', 'val', 'test'] if 'all' in args.splits else args.splits

    print('images root: {}'.format(images_root))
    print('labels root: {}'.format(labels_root))
    print('output root: {}'.format(out_root))
    if class_map is None:
        print('class mapping: output_id = yolo_id + {}'.format(args.class_offset))
    else:
        print('class mapping: {}'.format(class_map))

    raw_total = Counter()
    mapped_total = Counter()
    total_written = 0
    for split in splits:
        raw_counter, mapped_counter, written = convert_split(
            images_root,
            labels_root,
            out_root,
            split,
            class_map,
            args.class_offset,
            copy_images=not args.reencode_images,
        )
        raw_total.update(raw_counter)
        mapped_total.update(mapped_counter)
        total_written += written

    write_class_config(out_root / 'classes.json', mapped_total, args.class_names)

    print('total written: {}'.format(total_written))
    print('raw YOLO class objects:')
    for class_id in sorted(raw_total.keys()):
        print('  {}: {}'.format(class_id, raw_total[class_id]))
    print('output mask class objects:')
    for class_id in sorted(mapped_total.keys()):
        print('  {}: {}'.format(class_id, mapped_total[class_id]))
    print('class config: {}'.format(out_root / 'classes.json'))


if __name__ == '__main__':
    main()
