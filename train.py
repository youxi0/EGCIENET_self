import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from baseline import Mnet
from lib.class_config import load_class_config
from lib.class_config import num_classes as config_num_classes
from lib.data_prefetcher import DataPrefetcher
from lib.dataset import Data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_iou_loss(pred, target, eps=1e-6):
    dims = (1, 2, 3)
    inter = torch.sum(pred * target, dim=dims)
    union = torch.sum(pred + target, dim=dims) - inter
    return 1.0 - torch.mean((inter + eps) / (union + eps))


def single_binary_loss(logit, label):
    logit = F.interpolate(logit, label.shape[2:], mode='bilinear', align_corners=True)
    prob = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, label, reduction='mean')
    return bce + soft_iou_loss(prob, label)


def foreground_dice_loss(logit, label, num_classes, ignore_index=255, eps=1e-6):
    valid = label != ignore_index
    safe_label = label.clone()
    safe_label[~valid] = 0
    safe_label = safe_label.clamp(0, num_classes - 1)

    prob = F.softmax(logit, dim=1)
    target = F.one_hot(safe_label, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    prob = prob * valid
    target = target * valid

    if num_classes > 1:
        prob = prob[:, 1:, :, :]
        target = target[:, 1:, :, :]

    inter = torch.sum(prob * target, dim=(0, 2, 3))
    denom = torch.sum(prob + target, dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def single_multiclass_loss(logit, label, num_classes, class_weights=None):
    logit = F.interpolate(logit, label.shape[-2:], mode='bilinear', align_corners=True)
    ce = F.cross_entropy(logit, label.long(), weight=class_weights, ignore_index=255)
    dice = foreground_dice_loss(logit, label.long(), num_classes)
    return ce + dice


def segmentation_loss(score1, score2, score3, label, task, num_classes, class_weights=None):
    if task == 'multiclass':
        return (
            single_multiclass_loss(score1, label, num_classes, class_weights)
            + single_multiclass_loss(score2, label, num_classes, class_weights)
            + single_multiclass_loss(score3, label, num_classes, class_weights)
        )

    return (
        single_binary_loss(score1, label)
        + single_binary_loss(score2, label)
        + single_binary_loss(score3, label)
    )


def edge_loss(edge_logit, edge_target):
    if edge_target.shape[2:] != edge_logit.shape[2:]:
        edge_target = F.interpolate(edge_target, edge_logit.shape[2:], mode='area')
    edge_target = edge_target.clamp(0, 1)
    return F.binary_cross_entropy_with_logits(edge_logit, edge_target, reduction='mean')


def resolve_pretrained(value):
    if value is None:
        return False

    value = str(value).strip()
    if value.lower() in ('', 'none', 'false', '0', 'no'):
        return False

    if not os.path.isfile(value):
        raise FileNotFoundError(
            "MiT-B3 pretrained weight not found: '{}'. "
            "Download the SegFormer MiT-B3 ImageNet-1K weight and pass "
            "--pretrained /path/to/mit_b3.pth, or pass --pretrained none "
            "to train from scratch.".format(value)
        )

    return value


def resolve_num_classes(task, class_config_path, num_classes_arg):
    if task == 'binary':
        return 1, None
    config = load_class_config(class_config_path)
    num_classes = int(num_classes_arg) if num_classes_arg > 0 else config_num_classes(config)
    if num_classes < 2:
        raise ValueError('multiclass training requires at least 2 classes.')
    return num_classes, config


def resolve_class_weights(values, num_classes):
    if values is None:
        return None
    if len(values) != num_classes:
        raise ValueError('--class-weights expects {} values, got {}'.format(num_classes, len(values)))
    return torch.tensor(values, dtype=torch.float32).cuda()


def checkpoint_payload(net, args, num_classes, class_config):
    payload = {
        'state_dict': net.state_dict(),
        'task': args.task,
        'num_classes': num_classes,
        'edge_channels': args.edge_channels,
        'image_size': args.image_size,
    }
    if class_config is not None:
        payload['class_config'] = class_config
        payload['class_config_path'] = args.class_config
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description='Train EGCIENet with a distilled edge branch.')
    parser.add_argument('--task', choices=('binary', 'multiclass'), default='binary')
    parser.add_argument('--train-root', default='./Dataset/AEBIS/Train/', help='Training dataset root.')
    parser.add_argument('--class-config', default='./Dataset/AEBIS_MultiClass/classes.json')
    parser.add_argument('--num-classes', type=int, default=0, help='Override number of classes for multiclass.')
    parser.add_argument(
        '--class-weights',
        type=float,
        nargs='*',
        default=None,
        help='Optional CE weights, one value per class. Example: --class-weights 0.2 1 1 1 1',
    )
    parser.add_argument('--save-path', default='./model', help='Directory for checkpoints.')
    parser.add_argument(
        '--pretrained',
        default='mit_b3.pth',
        help="MiT pretrained weight path. Use 'none' to train from scratch.",
    )
    parser.add_argument('--image-size', type=int, default=352)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay-epochs', type=int, nargs='*', default=[60, 80])
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--edge-loss-weight', type=float, default=1.0)
    parser.add_argument('--edge-channels', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=118)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--amp', action='store_true', help='Use CUDA AMP mixed precision.')
    parser.add_argument(
        '--use-teacher-edge-in-seg',
        action='store_true',
        help='Use SAM edge as segmentation guidance during training. Default uses predicted edge.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.save_path, exist_ok=True)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    num_classes, class_config = resolve_num_classes(args.task, args.class_config, args.num_classes)
    class_weights = resolve_class_weights(args.class_weights, num_classes)

    dataset = Data(
        args.train_root,
        mode='train',
        image_size=args.image_size,
        require_edge=True,
        task=args.task,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    pretrained = resolve_pretrained(args.pretrained)
    net = Mnet(pretrained=pretrained, edge_channels=args.edge_channels, num_classes=num_classes).cuda()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    num_params = sum(p.numel() for p in net.parameters())
    print('task: {}, num_classes: {}'.format(args.task, num_classes))
    print('params: {:.2f}M'.format(num_params / 1e6))
    print('train samples: {}, iters/epoch: {}'.format(len(dataset), len(loader)))

    lr = args.lr
    net.train()
    for epochi in range(1, args.epoch + 1):
        if epochi in args.lr_decay_epochs:
            lr = lr / 10.0
            for group in optimizer.param_groups:
                group['lr'] = lr
            print('lr decayed to {}'.format(lr))

        prefetcher = DataPrefetcher(loader)
        rgb, label, edge = prefetcher.next()
        running_total = 0.0
        running_seg = 0.0
        running_edge = 0.0
        i = 0

        while rgb is not None:
            i += 1
            if args.task == 'multiclass':
                label = label.long()
            else:
                label = label.float().clamp(0, 1)
            edge = edge.float().clamp(0, 1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                score1, score2, score3, _, _, _, edge_logit, _ = net(
                    rgb,
                    edge=edge,
                    use_teacher_edge=args.use_teacher_edge_in_seg,
                )
                seg_loss = segmentation_loss(
                    score1,
                    score2,
                    score3,
                    label,
                    args.task,
                    num_classes,
                    class_weights,
                )
                e_loss = edge_loss(edge_logit, edge)
                total_loss = seg_loss + args.edge_loss_weight * e_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_total += total_loss.item()
            running_seg += seg_loss.item()
            running_edge += e_loss.item()

            if i % args.print_freq == 0:
                denom = float(args.print_freq)
                print(
                    'epoch: [{:3d}/{:3d}], iter: [{:5d}/{:5d}] || '
                    'loss: {:.4f} seg: {:.4f} edge: {:.4f} || lr: {:.6f}'.format(
                        epochi,
                        args.epoch,
                        i,
                        len(loader),
                        running_total / denom,
                        running_seg / denom,
                        running_edge / denom,
                        lr,
                    )
                )
                running_total = 0.0
                running_seg = 0.0
                running_edge = 0.0

            rgb, label, edge = prefetcher.next()

        if epochi >= 25 and epochi % 25 == 0:
            torch.save(
                checkpoint_payload(net, args, num_classes, class_config),
                os.path.join(args.save_path, 'epoch_{}.pth'.format(epochi)),
            )

    torch.save(
        checkpoint_payload(net, args, num_classes, class_config),
        os.path.join(args.save_path, 'final.pth'),
    )
