#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_gan.py - 使用改进的 UNet + GAN 去雾模型进行推理（已适配新架构）
"""

import os
import argparse
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from DGD_UNet_GAN import Generator  # 新模型无 bilinear 参数

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def gaussian_window_2d(size: int, sigma: float = None) -> torch.Tensor:
    """返回形状 (1,1,size,size) 的高斯权重张量（CPU）"""
    if sigma is None:
        sigma = size / 6.0
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    window = gauss[:, None] * gauss[None, :]
    return (window / window.max()).view(1, 1, size, size)


def load_depth_tensor(depth_dir: str, img_name: str, target_size: tuple, device: torch.device) -> torch.Tensor:
    """加载与 img_name 对应的深度图，返回 [1,1,H,W] 的张量（已移至 device）"""
    if depth_dir is None:
        return torch.zeros(1, 1, *target_size, device=device)

    depth_path = Path(depth_dir) / img_name
    # 尝试常见扩展名
    if not depth_path.exists():
        stem = depth_path.stem
        for ext in ['.png', '.npy', '.jpg', '.jpeg', '.bmp']:
            candidate = depth_path.with_suffix(ext)
            if candidate.exists():
                depth_path = candidate
                break
        else:
            logger.warning("Depth image for %s not found, using zeros.", img_name)
            return torch.zeros(1, 1, *target_size, device=device)

    try:
        if depth_path.suffix == '.npy':
            depth_np = np.load(str(depth_path)).astype(np.float32)
        else:
            depth_np = np.array(Image.open(depth_path).convert('L')).astype(np.float32)
    except Exception as e:
        logger.error("Failed to load depth image %s: %s", depth_path, e)
        return torch.zeros(1, 1, *target_size, device=device)

    # 归一化到 [0,1]
    d_min, d_max = depth_np.min(), depth_np.max()
    depth_np = (depth_np - d_min) / (d_max - d_min + 1e-6)

    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(device)
    if depth_tensor.shape[2:] != target_size:
        depth_tensor = F.interpolate(depth_tensor, size=target_size, mode='bilinear', align_corners=False)
    return depth_tensor


def process_single_image(generator: torch.nn.Module,
                         img_path: str,
                         depth_dir: str,
                         output_dir: str,
                         patch_size: int,
                         overlap: int,
                         sigma: float,
                         device: torch.device):
    """处理单张图像并保存结果"""
    img_name = Path(img_path).name
    logger.info("Processing %s ...", img_name)

    # 加载 RGB 图像并预处理到 [-1,1]
    haze_rgb = Image.open(img_path).convert('RGB')
    rgb_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    rgb_tensor = rgb_transform(haze_rgb).unsqueeze(0).to(device)  # [1,3,H,W]

    # 加载深度图
    _, _, H, W = rgb_tensor.shape
    depth_tensor = load_depth_tensor(depth_dir, img_name, (H, W), device)

    # 拼接为 4 通道输入
    input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=1)  # [1,4,H,W]

    # 高斯窗口
    gauss_weights = gaussian_window_2d(patch_size, sigma).to(device)

    # 分块推理
    with torch.inference_mode():
        if H <= patch_size and W <= patch_size:
            pad_h = max(0, patch_size - H)
            pad_w = max(0, patch_size - W)
            if pad_h > 0 or pad_w > 0:
                input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='replicate')
            output = generator(input_tensor)[:, :, :H, :W]
        else:
            stride = patch_size - overlap
            num_h = (H - patch_size + stride - 1) // stride + 1
            num_w = (W - patch_size + stride - 1) // stride + 1

            output_accum = torch.zeros(1, 3, H, W, device=device)
            weight_accum = torch.zeros(1, 1, H, W, device=device)

            for i in range(num_h):
                for j in range(num_w):
                    sh = i * stride
                    sw = j * stride
                    eh = min(sh + patch_size, H)
                    ew = min(sw + patch_size, W)
                    ph = patch_size - (eh - sh)
                    pw = patch_size - (ew - sw)

                    patch = input_tensor[:, :, sh:eh, sw:ew]
                    if ph > 0 or pw > 0:
                        patch = F.pad(patch, (0, pw, 0, ph), mode='replicate')

                    res = generator(patch)                    # [1,3,patch_size,patch_size]
                    res = res[:, :, :eh-sh, :ew-sw]          # 裁剪到实际区域

                    w = gauss_weights[:, :, :res.shape[2], :res.shape[3]]
                    output_accum[:, :, sh:eh, sw:ew] += res * w
                    weight_accum[:, :, sh:eh, sw:ew] += w

            output = output_accum / torch.clamp(weight_accum, min=1e-8)

    # 转回 CPU 并保存
    output = (output.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    out_pil = T.ToPILImage()(output)
    out_path = Path(output_dir) / f"dehazed_{img_name}"
    out_pil.save(str(out_path))
    logger.info("Saved: %s", out_path)


def infer(weight_path: str,
          input_dir: str,
          output_dir: str,
          depth_dir: str = None,
          patch_size: int = 256,
          overlap: int = 32,
          sigma: float = None):
    """主推理函数（已移除已废弃的 bilinear 参数）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    # 加载模型（新 Generator 无 bilinear 参数）
    generator = Generator().to(device)
    state_dict = torch.load(weight_path, map_location=device)

    # 修复 torch.compile 前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    generator.load_state_dict(new_state_dict)

    generator.eval()

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 支持的图像扩展名
    img_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

    for img_path in Path(input_dir).iterdir():
        if img_path.suffix.lower() not in img_extensions:
            continue
        try:
            process_single_image(
                generator, str(img_path), depth_dir, output_dir,
                patch_size, overlap, sigma, device
            )
        except Exception as e:
            logger.error("Failed to process %s: %s", img_path.name, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dehaze images using a GAN generator (optionally with depth guidance)")
    parser.add_argument('-w', '--weight_path', type=str, default='checkpoints_gan/best_generator.pth')
    parser.add_argument('-i', '--input_dir', type=str, default='test/samples')
    parser.add_argument('-o', '--output_dir', type=str, default='test/results_gan')
    parser.add_argument('-d', '--depth_dir', type=str, default=None,
                        help='Directory containing depth maps (optional). Expected to have same filename as input image.')
    # 已移除 no-gpu, bilinear 等旧参数
    parser.add_argument('-s', '--patch_size', type=int, default=256, help='Patch size for tiled inference')
    parser.add_argument('-v', '--overlap', type=int, default=32, help='Overlap between patches')
    parser.add_argument('--sigma', type=float, default=None, help='Sigma for Gaussian blending window (default: patch_size/6)')
    args = parser.parse_args()

    infer(args.weight_path, args.input_dir, args.output_dir, args.depth_dir,
          args.patch_size, args.overlap, args.sigma)
