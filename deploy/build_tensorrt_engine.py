import argparse
import os

import cv2
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    trt = None


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
MEAN_BGR = np.array([0.551, 0.619, 0.532], dtype=np.float32) * 255.0
STD_BGR = np.array([0.241, 0.236, 0.244], dtype=np.float32) * 255.0


def collect_images(root, limit=0):
    if os.path.isdir(os.path.join(root, 'JPEGImages')):
        root = os.path.join(root, 'JPEGImages')

    images = []
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTS:
                images.append(os.path.join(dirpath, filename))

    images = sorted(images)
    if limit and limit > 0:
        images = images[:limit]
    return images


def preprocess_bgr(image_path, image_size):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError('Could not read calibration image: {}'.format(image_path))
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = (image - MEAN_BGR) / STD_BGR
    image = image.transpose(2, 0, 1)
    return image


class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2 if trt is not None else object):
    def __init__(self, image_paths, batch_size, image_size, cache_file):
        if trt is None:
            raise ImportError('TensorRT Python package is not installed.')

        trt.IInt8EntropyCalibrator2.__init__(self)
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda

        self.cuda = cuda
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.cache_file = cache_file
        self.index = 0
        self.batch = np.zeros((batch_size, 3, image_size, image_size), dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.batch.nbytes)

        if not self.image_paths:
            raise ValueError('INT8 calibration needs at least one image.')
        print('INT8 calibration images: {}'.format(len(self.image_paths)))

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index + self.batch_size > len(self.image_paths):
            return None

        for i in range(self.batch_size):
            self.batch[i] = preprocess_bgr(self.image_paths[self.index + i], self.image_size)

        self.index += self.batch_size
        self.cuda.memcpy_htod(self.device_input, np.ascontiguousarray(self.batch))
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cache = f.read()
            print('loaded calibration cache: {}'.format(self.cache_file))
            return cache
        return None

    def write_calibration_cache(self, cache):
        if not self.cache_file:
            return
        directory = os.path.dirname(os.path.abspath(self.cache_file))
        os.makedirs(directory, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print('saved calibration cache: {}'.format(self.cache_file))


def parse_shape(value):
    if not value:
        return None
    parts = value.lower().replace(',', 'x').split('x')
    shape = tuple(int(part) for part in parts if part)
    if len(shape) != 4:
        raise ValueError('shape must be NCHW, e.g. 1x3x352x352: {}'.format(value))
    return shape


def set_workspace(config, workspace_mib):
    workspace_bytes = int(workspace_mib) << 20
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def print_network_io(network):
    print('network inputs:')
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print('  {} {}'.format(tensor.name, tuple(tensor.shape)))
    print('network outputs:')
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print('  {} {}'.format(tensor.name, tuple(tensor.shape)))


def parse_args():
    parser = argparse.ArgumentParser(description='Build TensorRT engine from EGCIENet ONNX.')
    parser.add_argument('--onnx', default='./deploy/egcienet_352.onnx', help='Input ONNX path.')
    parser.add_argument('--engine', default='./deploy/egcienet_352_fp16.engine', help='Output TensorRT engine path.')
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16', 'int8'])
    parser.add_argument('--workspace', type=int, default=2048, help='Workspace size in MiB.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=352)
    parser.add_argument('--min-shape', default='', help='Dynamic input min shape, e.g. 1x3x352x352.')
    parser.add_argument('--opt-shape', default='', help='Dynamic input opt shape, e.g. 1x3x352x352.')
    parser.add_argument('--max-shape', default='', help='Dynamic input max shape, e.g. 1x3x352x352.')
    parser.add_argument('--calib-root', default='./Dataset/AEBIS/Train/JPEGImages', help='Calibration image root.')
    parser.add_argument('--calib-cache', default='./deploy/int8_calib.cache')
    parser.add_argument('--num-calib', type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    if trt is None:
        raise ImportError('TensorRT Python package is not installed. Run this script on the Jetson/TensorRT machine.')

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError('Failed to parse ONNX: {}'.format(args.onnx))

    print_network_io(network)
    config = builder.create_builder_config()
    set_workspace(config, args.workspace)

    if args.precision == 'fp16':
        if not builder.platform_has_fast_fp16:
            print('warning: platform_has_fast_fp16 is False, but FP16 flag will still be set.')
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.precision == 'int8':
        if not builder.platform_has_fast_int8:
            print('warning: platform_has_fast_int8 is False, but INT8 flag will still be set.')
        config.set_flag(trt.BuilderFlag.INT8)
        image_paths = collect_images(args.calib_root, args.num_calib)
        config.int8_calibrator = ImageEntropyCalibrator(
            image_paths=image_paths,
            batch_size=args.batch_size,
            image_size=args.image_size,
            cache_file=args.calib_cache,
        )

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = tuple(input_tensor.shape)
    static_shape = (args.batch_size, 3, args.image_size, args.image_size)

    if any(dim < 0 for dim in input_shape):
        min_shape = parse_shape(args.min_shape) or static_shape
        opt_shape = parse_shape(args.opt_shape) or static_shape
        max_shape = parse_shape(args.max_shape) or static_shape
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        if args.precision == 'int8' and hasattr(config, 'set_calibration_profile'):
            config.set_calibration_profile(profile)
        print('dynamic profile {} min={} opt={} max={}'.format(input_name, min_shape, opt_shape, max_shape))
    elif input_shape != static_shape:
        print('warning: ONNX input shape {} differs from requested {}'.format(input_shape, static_shape))

    print('building {} engine...'.format(args.precision.upper()))
    if hasattr(builder, 'build_serialized_network'):
        serialized_engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        serialized_engine = engine.serialize() if engine is not None else None

    if serialized_engine is None:
        raise RuntimeError('TensorRT engine build failed.')

    os.makedirs(os.path.dirname(os.path.abspath(args.engine)), exist_ok=True)
    with open(args.engine, 'wb') as f:
        f.write(serialized_engine)
    print('saved engine: {}'.format(args.engine))


if __name__ == '__main__':
    main()
