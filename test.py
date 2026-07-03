import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline import Mnet
from lib.dataset import Data
from pytorch_iou.IOU_CrossValidation import miou


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
    return parser.parse_args()


def to_int(value):
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.out_path, exist_ok=True)

    edge_out_path = args.out_path.rstrip('/\\') + '_edge'
    if args.save_edge:
        os.makedirs(edge_out_path, exist_ok=True)

    data = Data(root=args.data_root, mode='test', image_size=args.image_size)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    net = Mnet(pretrained=False, edge_channels=args.edge_channels).cuda()
    print('loading model from {}...'.format(args.model_path))
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()

    img_num = len(loader)
    time_s = time.time()
    with torch.no_grad():
        for rgb, (h, w), name in loader:
            h = to_int(h)
            w = to_int(w)
            rgb = rgb.cuda().float()

            score1, score2, score3, s1, s2, s3, edge_logit, edge_prob = net(rgb)
            score1 = F.interpolate(score1, size=(h, w), mode='bilinear', align_corners=True)

            pred = np.squeeze(torch.sigmoid(score1).cpu().data.numpy())
            pred = (pred > args.threshold).astype(np.uint8) * 255
            mask_name = os.path.splitext(name[0])[0] + '.png'
            cv2.imwrite(os.path.join(args.out_path, mask_name), pred)

            if args.save_edge:
                edge = F.interpolate(edge_prob, size=(h, w), mode='bilinear', align_corners=True)
                edge = np.squeeze(edge.cpu().data.numpy())
                edge = np.clip(edge * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(edge_out_path, mask_name), edge)

            print('{} Done!'.format(name[0]))

    time_e = time.time()
    print('speed: {:.6f} FPS'.format(img_num / (time_e - time_s)))

    gt_dir = os.path.join(args.data_root, 'BlackWhite')
    if os.path.isdir(gt_dir):
        mIOU = miou(args.out_path, gt_dir)
        print('mIoU: {}'.format(mIOU))
