import argparse
import base64
import json
import os
import sys
import zipfile
from collections import Counter

import cv2
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.class_config import raw_label_to_class_id, write_class_config  # noqa: E402


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_split(split_file):
    if not split_file or not os.path.exists(split_file):
        return []
    names = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(stem(line))
    return names


def find_file_by_stem(directory, file_stem, exts=IMAGE_EXTS):
    if not directory or not os.path.isdir(directory):
        return None
    for ext in exts:
        path = os.path.join(directory, file_stem + ext)
        if os.path.exists(path):
            return path
    return None


def decode_labelme_image(data, json_dir='', image_root=''):
    image_data = data.get('imageData')
    if image_data:
        raw = base64.b64decode(image_data)
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is not None:
            return image

    candidates = []
    image_path = data.get('imagePath')
    if image_path:
        candidates.append(os.path.join(json_dir, image_path))
        if image_root:
            candidates.append(os.path.join(image_root, os.path.basename(image_path)))

    for path in candidates:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            return image

    raise FileNotFoundError('Could not decode Labelme image for {}'.format(image_path))


def polygon_to_mask(data, class_map):
    height = int(data['imageHeight'])
    width = int(data['imageWidth'])
    mask = np.zeros((height, width), dtype=np.uint8)
    binary = np.zeros((height, width), dtype=np.uint8)
    label_counter = Counter()

    for shape in data.get('shapes', []):
        label = str(shape.get('label', '')).strip()
        if label not in class_map:
            continue
        points = np.asarray(shape.get('points', []), dtype=np.float32)
        if len(points) < 3:
            continue
        points = np.round(points).astype(np.int32)
        class_id = int(class_map[label])
        cv2.fillPoly(mask, [points], class_id)
        cv2.fillPoly(binary, [points], 255)
        label_counter[label] += 1

    return mask, binary, label_counter


def fallback_edge_from_binary(binary):
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return cv2.subtract(dilated, eroded)


def list_labelme_jsons(path):
    if not os.path.exists(path):
        raise FileNotFoundError('Labelme path not found: {}'.format(path))

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith('.json'):
                    raw = zf.read(name)
                    yield name, json.loads(raw.decode('utf-8')), ''
        return

    for root, _, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith('.json'):
                json_path = os.path.join(root, filename)
                with open(json_path, 'r', encoding='utf-8') as f:
                    yield json_path, json.load(f), root


def prepare_output_dirs(root, split_name):
    split_root = os.path.join(root, split_name)
    dirs = {
        'images': os.path.join(split_root, 'JPEGImages'),
        'seg': os.path.join(split_root, 'SegClass'),
        'binary': os.path.join(split_root, 'BlackWhite'),
        'edge': os.path.join(split_root, 'Edge'),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def copy_or_make_edge(file_stem, binary, source_edge_dir, out_edge_dir, generate_missing_edge=False):
    edge_path = find_file_by_stem(source_edge_dir, file_stem)
    out_path = os.path.join(out_edge_dir, file_stem + '.png')
    if edge_path:
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if edge is not None:
            cv2.imwrite(out_path, edge)
            return True
    if generate_missing_edge:
        cv2.imwrite(out_path, fallback_edge_from_binary(binary))
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Labelme AEBIS_Class annotations to merged multiclass masks.')
    parser.add_argument('--labelme-root', default='./Dataset/AEBIS_Class', help='Labelme JSON directory or zip.')
    parser.add_argument('--binary-root', default='./Dataset/AEBIS', help='Original AEBIS root with Train/Test splits.')
    parser.add_argument('--out-root', default='./Dataset/AEBIS_MultiClass', help='Output multiclass dataset root.')
    parser.add_argument('--image-root', default='', help='Optional image root when JSON imageData is empty.')
    parser.add_argument('--config-out', default='', help='Output classes.json path. Default: out-root/classes.json.')
    parser.add_argument(
        '--generate-missing-edge',
        action='store_true',
        help='Generate an edge map from the segmentation mask when the source edge teacher is missing.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config_out or os.path.join(args.out_root, 'classes.json')
    config = write_class_config(config_path)
    class_map = raw_label_to_class_id(config)

    train_stems = set(read_split(os.path.join(args.binary_root, 'Train', 'train.txt')))
    test_stems = set(read_split(os.path.join(args.binary_root, 'Test', 'test.txt')))
    split_dirs = {
        'Train': prepare_output_dirs(args.out_root, 'Train'),
        'Test': prepare_output_dirs(args.out_root, 'Test'),
    }

    json_by_stem = {}
    json_dir_by_stem = {}
    for json_name, data, json_dir in list_labelme_jsons(args.labelme_root):
        file_stem = stem(json_name)
        json_by_stem[file_stem] = data
        json_dir_by_stem[file_stem] = json_dir

    label_counter = Counter()
    written = Counter()
    written_stems = {'Train': [], 'Test': []}
    missing = []
    missing_edge = []

    for split_name, stems in (('Train', train_stems), ('Test', test_stems)):
        source_edge_dir = os.path.join(args.binary_root, split_name, 'Edge')
        dirs = split_dirs[split_name]

        for file_stem in sorted(stems, key=lambda value: int(value) if value.isdigit() else value):
            data = json_by_stem.get(file_stem)
            if data is None:
                missing.append('{}:{}'.format(split_name, file_stem))
                continue

            image = decode_labelme_image(data, json_dir_by_stem.get(file_stem, ''), args.image_root)
            seg, binary, labels = polygon_to_mask(data, class_map)
            if not copy_or_make_edge(
                file_stem,
                binary,
                source_edge_dir,
                dirs['edge'],
                generate_missing_edge=args.generate_missing_edge,
            ):
                missing_edge.append('{}:{}'.format(split_name, file_stem))
                continue

            label_counter.update(labels)

            cv2.imwrite(os.path.join(dirs['images'], file_stem + '.jpg'), image)
            cv2.imwrite(os.path.join(dirs['seg'], file_stem + '.png'), seg)
            cv2.imwrite(os.path.join(dirs['binary'], file_stem + '.png'), binary)
            written[split_name] += 1
            written_stems[split_name].append(file_stem)

        split_txt_dst = os.path.join(args.out_root, split_name, split_name.lower() + '.txt')
        with open(split_txt_dst, 'w', encoding='utf-8') as f:
            for file_stem in written_stems[split_name]:
                f.write(file_stem + '.jpg\n')

    print('class config: {}'.format(config_path))
    print('written Train: {}'.format(written['Train']))
    print('written Test: {}'.format(written['Test']))
    print('raw label polygons:')
    for label, count in label_counter.most_common():
        print('  {}: {}'.format(label, count))
    if missing:
        print('missing JSON annotations: {}'.format(len(missing)))
        print('  {}'.format(', '.join(missing[:20])))
    if missing_edge:
        print('missing edge teachers: {}'.format(len(missing_edge)))
        print('  {}'.format(', '.join(missing_edge[:20])))


if __name__ == '__main__':
    main()
