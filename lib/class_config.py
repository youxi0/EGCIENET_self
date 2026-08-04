import json
import os


SIMILARITY_MERGED_CLASSES = [
    {
        'id': 0,
        'name': 'background',
        'raw_labels': [],
        'description': 'Non-defect background.',
    },
    {
        'id': 1,
        'name': 'burn',
        'raw_labels': ['Burn'],
        'description': 'Thermal discoloration or burn-like surface damage.',
    },
    {
        'id': 2,
        'name': 'crack_tear',
        'raw_labels': ['Crack', 'Tears'],
        'description': 'Linear fracture, crack, or tear-like discontinuity.',
    },
    {
        'id': 3,
        'name': 'material_loss',
        'raw_labels': ['Material missing', 'Nick'],
        'description': 'Material-loss, notch, or missing-edge damage.',
    },
    {
        'id': 4,
        'name': 'deformation',
        'raw_labels': ['Dent', 'Tip curl'],
        'description': 'Plastic deformation, dent, or curled-tip geometry change.',
    },
]


def default_class_config():
    return {
        'version': 1,
        'background_id': 0,
        'merge_strategy': 'similarity_by_damage_morphology',
        'classes': SIMILARITY_MERGED_CLASSES,
    }


def write_class_config(path, extra=None):
    config = default_class_config()
    if extra:
        config.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config


def load_class_config(path=None):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default_class_config()


def raw_label_to_class_id(config):
    mapping = {}
    for item in config['classes']:
        for label in item.get('raw_labels', []):
            mapping[label] = int(item['id'])
    return mapping


def class_names(config):
    classes = sorted(config['classes'], key=lambda item: int(item['id']))
    return [item['name'] for item in classes]


def num_classes(config):
    return max(int(item['id']) for item in config['classes']) + 1
