import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baseline import Mnet  # noqa: E402


class DeployModel(nn.Module):
    """Inference wrapper for ONNX/TensorRT deployment.

    The training model returns several deep-supervision tensors. Deployment only
    needs a full-resolution defect mask probability, plus an optional edge map
    for visualization/debugging.
    """

    def __init__(self, model, output_edge=False, output_logits=False):
        super(DeployModel, self).__init__()
        self.model = model
        self.output_edge = output_edge
        self.output_logits = output_logits

    def forward(self, image):
        score1, _, _, _, _, _, edge_logit, edge_prob = self.model(image)
        mask_logit = F.interpolate(score1, size=image.shape[2:], mode='bilinear', align_corners=True)

        if self.output_logits:
            mask = mask_logit
            edge = F.interpolate(edge_logit, size=image.shape[2:], mode='bilinear', align_corners=True)
        else:
            mask = torch.sigmoid(mask_logit)
            edge = F.interpolate(edge_prob, size=image.shape[2:], mode='bilinear', align_corners=True)

        if self.output_edge:
            return mask, edge
        return mask


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def load_mnet(model_path, edge_channels=32, device='cpu', strict=True):
    model = Mnet(pretrained=False, edge_channels=edge_channels)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = strip_module_prefix(checkpoint)

    if strict:
        model.load_state_dict(checkpoint, strict=True)
    else:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing:
            print('missing keys: {}'.format(len(missing)))
        if unexpected:
            print('unexpected keys: {}'.format(len(unexpected)))

    model.to(device)
    model.eval()
    return model


def inspect_onnx(onnx_path):
    try:
        import onnx
    except ImportError:
        print('onnx package is not installed; skipped ONNX checker.')
        return

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print('ONNX checker: OK')

    print('inputs:')
    for item in model.graph.input:
        dims = []
        for dim in item.type.tensor_type.shape.dim:
            dims.append(dim.dim_param if dim.dim_param else dim.dim_value)
        print('  {} {}'.format(item.name, dims))

    print('outputs:')
    for item in model.graph.output:
        dims = []
        for dim in item.type.tensor_type.shape.dim:
            dims.append(dim.dim_param if dim.dim_param else dim.dim_value)
        print('  {} {}'.format(item.name, dims))


def parse_args():
    parser = argparse.ArgumentParser(description='Export EGCIENet to ONNX for deployment.')
    parser.add_argument('--model-path', default='./model/final.pth', help='PyTorch checkpoint path.')
    parser.add_argument('--onnx-path', default='./deploy/egcienet_352.onnx', help='Output ONNX path.')
    parser.add_argument('--image-size', type=int, default=352, help='Export input height/width.')
    parser.add_argument('--batch-size', type=int, default=1, help='Export batch size.')
    parser.add_argument('--edge-channels', type=int, default=32)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--opset', type=int, default=13)
    parser.add_argument('--output-edge', action='store_true', help='Export edge probability as a second output.')
    parser.add_argument('--output-logits', action='store_true', help='Export logits instead of probabilities.')
    parser.add_argument('--dynamic', action='store_true', help='Use dynamic batch/height/width axes.')
    parser.add_argument('--non-strict', action='store_true', help='Load checkpoint with strict=False.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is unavailable, exporting on CPU.')
        args.device = 'cpu'

    os.makedirs(os.path.dirname(os.path.abspath(args.onnx_path)), exist_ok=True)

    base_model = load_mnet(
        args.model_path,
        edge_channels=args.edge_channels,
        device=args.device,
        strict=not args.non_strict,
    )
    deploy_model = DeployModel(
        base_model,
        output_edge=args.output_edge,
        output_logits=args.output_logits,
    ).to(args.device)
    deploy_model.eval()

    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=args.device)
    output_names = ['mask_logit' if args.output_logits else 'mask']
    if args.output_edge:
        output_names.append('edge_logit' if args.output_logits else 'edge')

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            'image': {0: 'batch', 2: 'height', 3: 'width'},
            output_names[0]: {0: 'batch', 2: 'height', 3: 'width'},
        }
        if args.output_edge:
            dynamic_axes[output_names[1]] = {0: 'batch', 2: 'height', 3: 'width'}

    with torch.no_grad():
        outputs = deploy_model(dummy)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    print('PyTorch deploy I/O:')
    print('  input image: {}'.format(tuple(dummy.shape)))
    for name, tensor in zip(output_names, outputs):
        print('  output {}: {}'.format(name, tuple(tensor.shape)))

    torch.onnx.export(
        deploy_model,
        dummy,
        args.onnx_path,
        input_names=['image'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print('exported ONNX: {}'.format(args.onnx_path))
    inspect_onnx(args.onnx_path)


if __name__ == '__main__':
    main()
