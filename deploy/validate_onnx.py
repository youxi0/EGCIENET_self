import argparse
import os
import sys

import cv2
import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deploy.export_onnx import DeployModel, load_mnet  # noqa: E402


MEAN_BGR = np.array([0.551, 0.619, 0.532], dtype=np.float32) * 255.0
STD_BGR = np.array([0.241, 0.236, 0.244], dtype=np.float32) * 255.0


def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError('Could not read image: {}'.format(image_path))
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = (image - MEAN_BGR) / STD_BGR
    image = image.transpose(2, 0, 1)
    return np.expand_dims(image, axis=0).astype(np.float32)


def make_input(image_path, image_size):
    if image_path:
        return preprocess_image(image_path, image_size)
    rng = np.random.default_rng(118)
    return rng.standard_normal((1, 3, image_size, image_size)).astype(np.float32)


def save_mask(path, mask):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mask = np.squeeze(mask)
    mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, mask)
    print('saved ONNX mask: {}'.format(path))


def parse_args():
    parser = argparse.ArgumentParser(description='Compare PyTorch and ONNX outputs.')
    parser.add_argument('--model-path', default='./model/final.pth')
    parser.add_argument('--onnx-path', default='./deploy/egcienet_352.onnx')
    parser.add_argument('--image-path', default='', help='Optional input image. Random input is used if empty.')
    parser.add_argument('--image-size', type=int, default=352)
    parser.add_argument('--edge-channels', type=int, default=32)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--save-mask', default='', help='Optional path to save ONNX mask probability as png.')
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError('onnxruntime is not installed. Install onnxruntime or onnxruntime-gpu first.')

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is unavailable, validating PyTorch on CPU.')
        args.device = 'cpu'

    input_np = make_input(args.image_path, args.image_size)

    base_model = load_mnet(args.model_path, edge_channels=args.edge_channels, device=args.device)
    torch_model = DeployModel(base_model, output_edge=False, output_logits=False).to(args.device)
    torch_model.eval()
    with torch.no_grad():
        torch_output = torch_model(torch.from_numpy(input_np).to(args.device)).cpu().numpy()

    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    session = ort.InferenceSession(args.onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: input_np})[0]

    diff = np.abs(torch_output - onnx_output)
    print('PyTorch output shape: {}'.format(torch_output.shape))
    print('ONNX output shape: {}'.format(onnx_output.shape))
    print('max abs diff: {:.8f}'.format(float(diff.max())))
    print('mean abs diff: {:.8f}'.format(float(diff.mean())))
    save_mask(args.save_mask, onnx_output)


if __name__ == '__main__':
    main()
