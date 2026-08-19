#!/usr/bin/env python3

"""
V14：在 V12 的基础上替换 block3.0～block3.17 的布局链、DWConv 和 GELU。

V14 与 V13 使用相同的 packed-weight Plugin 和 [B,H*W,C] 接口，但设置
fuse_gelu=1，并把原 exact-erf GELU 的最终输出 tensor 交给 Plugin 直接产生。

注意：当前 CUDA Plugin 内部使用 FP32 中间值计算 tanh 近似 GELU，然后舍入为
FP16；它不是原 ONNX 的 exact-erf GELU。该版本必须单独进行完整精度验证。
"""

import argparse
from pathlib import Path

import replace_block3_layout_chain_with_packed_dwconv_v13 as v13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V14: replace block3 layout/DWConv/GELU chains with fused plugins"
        ),
    )
    parser.add_argument(
        "--input",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v12_"
            "block1_packed_dwconv_gelu.onnx"
        ),
        help="validated V12 ONNX containing three block1 plugins",
    )
    parser.add_argument(
        "--output",
        default=(
            "deploy/egcienet_352_multiclass_qdq_v14_"
            "block3_packed_dwconv_gelu.onnx"
        ),
        help="V14 ONNX with 18 block3 packed DWConv + GELU plugins",
    )
    parser.add_argument("--height", type=int, default=v13.DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=v13.DEFAULT_WIDTH)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v13.generate_variant(
        Path(args.input),
        Path(args.output),
        args.height,
        args.width,
        fuse_gelu=True,
        force=args.force,
    )


if __name__ == "__main__":
    main()
