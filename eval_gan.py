#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_gan.py - 评估去雾GAN模型在测试集上的客观指标，保存CSV结果及恢复图像
"""

import os
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from DGD_UNet_GAN import Generator

# 尝试导入指标库
try:
    from skimage.metrics import structural_similarity as ssim_skimage
except ImportError:
    ssim_skimage = None

try:
    import lpips
    lpips_available = True
except ImportError:
    lpips_available = False

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    # 自定义PSNR，但为了SSIM只使用StructuralSimilarityIndexMeasure
    torchmetrics_available = True
except ImportError:
    torchmetrics_available = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestDataset(Dataset):
    """测试数据集，加载 haze、clear 和可选的 depth，并缩放到指定尺寸"""

    def __init__(self, haze_dir: str, clear_dir: str, depth_dir: str = None, size: int = 256):
        self.haze_paths = sorted([p for p in Path(haze_dir).iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg','.bmp')])
        self.clear_paths = sorted([p for p in Path(clear_dir).iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg','.bmp')])
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.size = size

        assert len(self.haze_paths) == len(self.clear_paths), \
            f"haze ({len(self.haze_paths)}) and clear ({len(self.clear_paths)}) directories have different number of files."

        self.filenames = [p.name for p in self.haze_paths]

        self.to_tensor = T.ToTensor()
        self.norm_rgb = T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

    def __len__(self):
        return len(self.haze_paths)

    def _load_depth(self, path: Path) -> torch.Tensor:
        if not path.exists():
            return torch.zeros(1, 1, self.size, self.size)
        try:
            if path.suffix == '.npy':
                depth = np.load(str(path)).astype(np.float32)
            else:
                depth = np.array(Image.open(path).convert('L')).astype(np.float32)
        except Exception as e:
            logger.warning("Failed to load depth %s: %s. Using zeros.", path, e)
            return torch.zeros(1, 1, self.size, self.size)
        d_min, d_max = depth.min(), depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth_t = F.interpolate(depth_t, size=(self.size, self.size), mode='bilinear', align_corners=False)
        return depth_t

    def __getitem__(self, idx):
        haze_path = self.haze_paths[idx]
        clear_path = self.clear_paths[idx]

        haze_img = Image.open(haze_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')

        haze_img = TF.resize(haze_img, (self.size, self.size))
        clear_img = TF.resize(clear_img, (self.size, self.size))

        haze_t = self.to_tensor(haze_img)
        clear_t = self.to_tensor(clear_img)
        haze_norm = self.norm_rgb(haze_t)
        clear_norm = self.norm_rgb(clear_t)

        if self.depth_dir is not None:
            depth_path = self.depth_dir / haze_path.name
            depth_t = self._load_depth(depth_path)
        else:
            depth_t = torch.zeros(1, 1, self.size, self.size)

        input_t = torch.cat([haze_norm, depth_t.squeeze(0)], dim=0)

        return {
            'filename': self.filenames[idx],
            'haze': input_t,
            'clear': clear_norm
        }


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, lpips_model=None):
    """
    计算 PSNR, SSIM, LPIPS。
    pred, target: [B,3,H,W] 在 [-1,1] 范围
    返回字典 {psnr, ssim, lpips}，每个值为 [B] 张量
    """
    # 转换到 [0,1] 范围用于 SSIM 和 LPIPS
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2

    # PSNR (使用 [-1,1] 范围，最大值为2)
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1,2,3])
    psnr = 10 * torch.log10(4.0 / (mse + 1e-8))

    # SSIM
    if torchmetrics_available:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
        ssim = ssim_metric(pred_01, target_01)
    elif ssim_skimage is not None:
        ssim_list = []
        for i in range(pred.size(0)):
            p = pred_01[i].cpu().numpy().transpose(1,2,0)
            t = target_01[i].cpu().numpy().transpose(1,2,0)
            s = ssim_skimage(p, t, channel_axis=-1, data_range=1.0)
            ssim_list.append(s)
        ssim = torch.tensor(ssim_list, device=pred.device)
    else:
        logger.warning("SSIM not available (install torchmetrics or skimage). Returning NaN.")
        ssim = torch.full_like(psnr, float('nan'))

    # LPIPS
    if lpips_model is not None:
        lpips_val = lpips_model(pred, target)
        lpips = lpips_val.squeeze()
    else:
        lpips = torch.full_like(psnr, float('nan'))

    return {
        'psnr': psnr.detach().cpu(),
        'ssim': ssim.detach().cpu(),
        'lpips': lpips.detach().cpu()
    }


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    # 加载生成器
    generator = Generator().to(device)
    state_dict = torch.load(args.weight_path, map_location=device)
    # 处理 torch.compile 前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    generator.load_state_dict(new_state_dict)
    generator.eval()
    logger.info("Generator loaded from %s", args.weight_path)

    # 创建测试数据集
    dataset = TestDataset(
        haze_dir=args.haze_dir,
        clear_dir=args.clear_dir,
        depth_dir=args.depth_dir,
        size=args.size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logger.info("Test dataset: %d samples", len(dataset))

    # 准备 LPIPS 模型（如果可用）
    lpips_model = None
    if args.compute_lpips and lpips_available:
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        logger.info("LPIPS model loaded (alex).")
    elif args.compute_lpips:
        logger.warning("LPIPS requested but lpips package not installed, skipping.")

    # 收集所有指标和文件名
    all_filenames = []
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_times = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            filenames = batch['filename']
            haze = batch['haze'].to(device, non_blocking=True)
            clear = batch['clear'].to(device, non_blocking=True)

            # 计时：生成器前向传播
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            pred = generator(haze)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            batch_time = t_end - t_start

            # 保存输出图像（如果指定了 --save_output）
            if args.save_output:
                output_dir = Path(args.save_output)
                output_dir.mkdir(parents=True, exist_ok=True)
                # 将输出从 [-1,1] 转换到 [0,1]，并截断防止溢出
                pred_img = torch.clamp((pred + 1) / 2, 0.0, 1.0)
                for i, fname in enumerate(filenames):
                    out_path = output_dir / fname
                    # ToPILImage 期望输入为 [C,H,W] 类型为 float 在 [0,1] 或 uint8
                    pil_img = T.ToPILImage()(pred_img[i].cpu())
                    pil_img.save(str(out_path))

            metrics = compute_metrics(pred, clear, lpips_model)

            batch_size_actual = len(filenames)
            per_img_time = batch_time / batch_size_actual

            all_filenames.extend(filenames)
            all_psnr.extend(metrics['psnr'].tolist())
            all_ssim.extend(metrics['ssim'].tolist())
            all_lpips.extend(metrics['lpips'].tolist())
            all_times.extend([per_img_time] * batch_size_actual)

            if (batch_idx + 1) % 20 == 0:
                logger.info("Batch %d/%d processed. Avg time/image: %.4f s",
                            batch_idx + 1, len(dataloader), per_img_time)

    # 转换为 numpy 数组用于统计
    psnr_arr = np.array(all_psnr)
    ssim_arr = np.array(all_ssim)
    lpips_arr = np.array(all_lpips)
    time_arr = np.array(all_times)

    # 计算平均值和标准差
    avg_psnr = np.mean(psnr_arr)
    std_psnr = np.std(psnr_arr)
    avg_ssim = np.mean(ssim_arr)
    std_ssim = np.std(ssim_arr)
    avg_lpips = np.mean(lpips_arr)
    std_lpips = np.std(lpips_arr)
    avg_time = np.mean(time_arr)
    std_time = np.std(time_arr)

    logger.info("=" * 60)
    logger.info("Evaluation Results on %d samples:", len(all_psnr))
    logger.info("  PSNR  : %.4f ± %.4f dB", avg_psnr, std_psnr)
    logger.info("  SSIM  : %.4f ± %.4f", avg_ssim, std_ssim)
    logger.info("  LPIPS : %.4f ± %.4f", avg_lpips, std_lpips)
    logger.info("  Time  : %.4f ± %.4f s per image", avg_time, std_time)
    logger.info("=" * 60)

    # 保存 CSV 文件
    if args.csv_path:
        df = pd.DataFrame({
            'filename': all_filenames,
            'psnr': all_psnr,
            'ssim': all_ssim,
            'lpips': all_lpips,
            'time_sec': all_times
        })

        # 添加平均和标准差行
        stats_df = pd.DataFrame([
            ['Mean', avg_psnr, avg_ssim, avg_lpips, avg_time],
            ['Std',  std_psnr, std_ssim, std_lpips, std_time]
        ], columns=df.columns)

        df = pd.concat([df, stats_df], ignore_index=True)

        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, float_format='%.6f')
        logger.info("Results saved to %s", csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate GAN-based dehazing model on test set")
    parser.add_argument('--weight_path', type=str, default='checkpoints_gan/best_generator.pth',
                        help='Path to generator weights')
    parser.add_argument('--haze_dir', type=str, required=True, help='Directory with hazy images')
    parser.add_argument('--clear_dir', type=str, required=True, help='Directory with clear images (ground truth)')
    parser.add_argument('--depth_dir', type=str, default=None,
                        help='Optional directory with depth maps (filenames must match haze)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--size', type=int, default=256, help='Resize images to this size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--compute_lpips', action='store_true',
                        help='Compute LPIPS metric (requires lpips package)')
    parser.add_argument('--csv_path', type=str, default='evaluation_results.csv',
                        help='Path to save CSV results')
    parser.add_argument('--save_output', type=str, default=None,
                        help='If provided, save dehazed images to this directory (filenames preserved)')
    args = parser.parse_args()

    evaluate(args)
