import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class DehazingDataset(Dataset):
    def __init__(self, haze_dir, depth_dir, clear_dir, size=256,
                 use_augmentation=True, expand_factor=2, seed=16):
        """
        Args:
            haze_dir, depth_dir, clear_dir: 路径，同前
            size: 输出尺寸
            use_augmentation: 是否使用随机亮度/对比度等增强
            expand_factor: 每个原始样本生成的变体数量
            seed: 随机种子，用于所有随机操作，确保可重复性
        """
        self.haze_paths = sorted([os.path.join(haze_dir, f) for f in os.listdir(haze_dir)
                                  if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))])
        self.depth_paths = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                                   if f.lower().endswith(('.png','.npy','.tif','.bmp'))])
        self.clear_paths = sorted([os.path.join(clear_dir, f) for f in os.listdir(clear_dir)
                                   if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))])
        self.size = size
        self.use_aug = use_augmentation
        self.expand_factor = expand_factor
        self.seed = seed   # 保存种子
        self.to_tensor = T.ToTensor()
        self.norm_rgb = T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

        # 原始样本数量
        self.base_len = min(len(self.haze_paths), len(self.depth_paths), len(self.clear_paths))

    def __len__(self):
        return self.base_len * self.expand_factor

    def _load_depth(self, path):
        """返回 float32 数组，值域 [0,1]"""
        if path.endswith('.npy'):
            depth = np.load(path).astype(np.float32)
        else:
            depth = np.array(Image.open(path).convert('L')).astype(np.float32)
        d_min, d_max = depth.min(), depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        return depth   # shape [H, W]

    def _transform_variant_0(self, haze, depth, clear):
        """变体0：直接缩放为 size×size（完全保留原图信息，不裁剪）"""
        haze = TF.resize(haze, (self.size, self.size))
        depth = TF.resize(depth, (self.size, self.size))
        clear = TF.resize(clear, (self.size, self.size))
        return haze, depth, clear

    def _transform_variant_random(self, haze, depth, clear, rng):
        """
        随机变换：使用独立的 random.Random 实例 rng 控制所有随机操作
        包括随机缩放/裁剪、翻转、旋转、亮度/对比度增强
        """
        # 随机选择 resize 或 crop
        # 随机裁剪位置（使用 rng 生成坐标）
        w, h = haze.size   # PIL 图像 (width, height)
        i = rng.randint(0, h - self.size)
        j = rng.randint(0, w - self.size)
        haze = TF.crop(haze, i, j, self.size, self.size)
        depth = TF.crop(depth, i, j, self.size, self.size)
        clear = TF.crop(clear, i, j, self.size, self.size)

        # 随机水平翻转
        if rng.random() > 0.5:
            haze = TF.hflip(haze)
            depth = TF.hflip(depth)
            clear = TF.hflip(clear)
        # 随机垂直翻转
        if rng.random() > 0.5:
            haze = TF.vflip(haze)
            depth = TF.vflip(depth)
            clear = TF.vflip(clear)

        # 随机旋转（0°, 90°, 180°, 270°）
        angle = rng.choice([0, 90, 180, 270])
        haze = TF.rotate(haze, angle)
        depth = TF.rotate(depth, angle)
        clear = TF.rotate(clear, angle)

        # 亮度/对比度增强（仅 haze）
        if self.use_aug:
            if rng.random() > 0.5:
                brightness = rng.uniform(0.7, 1.3)
                contrast = rng.uniform(0.7, 1.3)
                haze = TF.adjust_brightness(haze, brightness)
                haze = TF.adjust_contrast(haze, contrast)

        return haze, depth, clear

    def __getitem__(self, idx):
        # 计算原始索引和变体序号
        base_idx = idx % self.base_len
        variant = idx // self.base_len   # 0, 1, ..., expand_factor-1

        # 加载原始图像
        haze_img = Image.open(self.haze_paths[base_idx]).convert('RGB')
        depth = self._load_depth(self.depth_paths[base_idx])
        depth_pil = Image.fromarray(depth, mode='F')   # 保留浮点
        clear_img = Image.open(self.clear_paths[base_idx]).convert('RGB')

        # 为当前样本创建独立的随机生成器（种子 = 全局种子 + 样本索引）
        rng = random.Random(self.seed + idx)

        if variant == 0:
            # 变体0：直接缩放为 size×size（无随机）
            haze_img, depth_pil, clear_img = self._transform_variant_0(haze_img, depth_pil, clear_img)
        else:
            # 其他变体：使用随机变换（基于 rng）
            haze_img, depth_pil, clear_img = self._transform_variant_random(
                haze_img, depth_pil, clear_img, rng
            )

        # 转换为 tensor 并归一化
        haze_tensor = self.to_tensor(haze_img)          # [3, H, W]
        depth_tensor = self.to_tensor(depth_pil)        # [1, H, W]
        clear_tensor = self.to_tensor(clear_img)        # [3, H, W]

        haze_rgb = self.norm_rgb(haze_tensor)
        haze_rgbd = torch.cat([haze_rgb, depth_tensor], dim=0)  # [4, H, W]
        clear_norm = self.norm_rgb(clear_tensor)

        return {
            'haze': haze_rgbd,
            'clear': clear_norm
        }
