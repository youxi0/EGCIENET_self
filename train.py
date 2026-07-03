import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from baseline import Mnet
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


def single_mask_loss(logit, label):
    logit = F.interpolate(logit, label.shape[2:], mode='bilinear', align_corners=True)
    prob = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, label, reduction='mean')
    return bce + soft_iou_loss(prob, label)


def segmentation_loss(score1, score2, score3, label):
    return (
        single_mask_loss(score1, label)
        + single_mask_loss(score2, label)
        + single_mask_loss(score3, label)
    )


def edge_loss(edge_logit, edge_target):
    edge_logit = F.interpolate(edge_logit, edge_target.shape[2:], mode='bilinear', align_corners=True)
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train EGCIENet with a distilled edge branch.')
    parser.add_argument('--train-root', default='./Dataset/AEBIS/Train/', help='Training dataset root.')
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
    parser.add_argument('--edge-channels', type=int, default=32)
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

    dataset = Data(args.train_root, mode='train', image_size=args.image_size, require_edge=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    pretrained = resolve_pretrained(args.pretrained)
    net = Mnet(pretrained=pretrained, edge_channels=args.edge_channels).cuda()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    num_params = sum(p.numel() for p in net.parameters())
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
            label = label.clamp(0, 1)
            edge = edge.clamp(0, 1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                score1, score2, score3, _, _, _, edge_logit, _ = net(
                    rgb,
                    edge=edge,
                    use_teacher_edge=args.use_teacher_edge_in_seg,
                )
                seg_loss = segmentation_loss(score1, score2, score3, label)
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
            torch.save(net.state_dict(), os.path.join(args.save_path, 'epoch_{}.pth'.format(epochi)))

    torch.save(net.state_dict(), os.path.join(args.save_path, 'final.pth'))
