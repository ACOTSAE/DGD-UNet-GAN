#!/bin/env python
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as T
from DGD_UNet_GAN import Generator, Discriminator
from dataset import DehazingDataset

torch.backends.cudnn.benchmark = True

# ========== 日志配置 ==========
def setup_logging(log_file: str = None, level=logging.INFO):
    """配置 logging，同时输出到控制台和文件（若指定 log_file）"""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除已有处理器，避免重复
    logger.handlers.clear()
    
    # 格式化器（高精度浮点数）
    formatter = logging.Formatter(
        fmt='%(message)s',
        datefmt=None
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定 log_file）
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# ========== 感知损失 ==========
class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.normalize = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_norm = self.normalize((pred+1)/2)
        target_norm = self.normalize((target+1)/2)
        pred_feat = self.vgg(pred_norm)
        target_feat = self.vgg(target_norm)
        return self.criterion(pred_feat, target_feat)

# ========== Hinge Loss 函数 ==========
def hinge_loss_d(real_pred, fake_pred):
    real_loss = torch.mean(F.relu(1 - real_pred))
    fake_loss = torch.mean(F.relu(1 + fake_pred))
    return (real_loss + fake_loss) * 0.5

def hinge_loss_g(fake_pred):
    return -torch.mean(fake_pred)

# ========== 检查点函数 ==========
def save_checkpoint(state, filename):
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer,
                    g_scheduler, d_scheduler, scaler, device):
    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    def fix_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    generator.load_state_dict(fix_state_dict(checkpoint['generator_state_dict']))
    discriminator.load_state_dict(fix_state_dict(checkpoint['discriminator_state_dict']))
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
    d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    logging.info(f"Resuming from epoch {start_epoch}, best val loss {best_val_loss:.6f}")
    return start_epoch, best_val_loss

# ========== 主训练函数 ==========
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    logging.info(f"Using device: {device}")

    # 混合精度开关：默认使用 FP16，可通过 --no-fp16 禁用
    use_amp = (not args.no_fp16) and torch.cuda.is_available()
    logging.info(f"Mixed precision (AMP): {use_amp}")

    # ---------- 数据集 ----------
    full_dataset = DehazingDataset(
        haze_dir=args.haze_dir,
        depth_dir=args.depth_dir,
        clear_dir=args.clear_dir,
        size=args.crop_size,
        seed=args.random_seed,
        expand_factor=args.expand_factor,
        use_augmentation=args.no_use_augmentation
    )
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ---------- 模型 ----------
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # ---------- 损失函数 ----------
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss(device)

    # ---------- 优化器 ----------
    g_optimizer = optim.AdamW(
        generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999), weight_decay=args.weight_decay
    )
    d_optimizer = optim.AdamW(
        discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999), weight_decay=args.weight_decay
    )

    # ---------- 学习率调度器（ReduceLROnPlateau） ----------
    #  g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #      g_optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    #  )
    #  d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #      d_optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    #  )
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer, T_max=args.epochs, eta_min=1e-6
    )
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        d_optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ---------- 混合精度缩放器 ----------
    scaler = GradScaler('cuda') if use_amp else None

    # ---------- 恢复训练 ----------
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = args.resume
    if checkpoint_path is None and args.auto_resume:
        auto_path = os.path.join(args.save_dir, 'last_checkpoint.pth')
        if os.path.exists(auto_path):
            checkpoint_path = auto_path
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path, generator, discriminator,
            g_optimizer, d_optimizer, g_scheduler, d_scheduler,
            scaler, device
        )
    else:
        logging.info("Starting training from scratch.")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- 编译生成器 ----------
    if hasattr(torch, 'compile') and args.compile:
        generator = torch.compile(generator)
        logging.info("Generator compiled with torch.compile")

    # ---------- 训练循环 ----------
    try:
        for epoch in range(start_epoch, args.epochs):
            generator.train()
            discriminator.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                haze = batch['haze'].to(device, non_blocking=True)   # [B,4,H,W]
                clear = batch['clear'].to(device, non_blocking=True) # [B,3,H,W]

                # ========== 训练判别器 ==========
                with torch.no_grad():
                    fake_clear_detached = generator(haze).detach()
                
                real_pred = discriminator(haze[:, :3], clear)
                fake_pred = discriminator(haze[:, :3], fake_clear_detached)

                d_optimizer.zero_grad()
                if use_amp:
                    with autocast(device_type='cuda'):
                        d_loss = hinge_loss_d(real_pred, fake_pred)
                else:
                    d_loss = hinge_loss_d(real_pred, fake_pred)
                
                if use_amp:
                    scaler.scale(d_loss).backward()
                    scaler.unscale_(d_optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    scaler.step(d_optimizer)
                else:
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    d_optimizer.step()

                # ========== 训练生成器 ==========
                g_optimizer.zero_grad()
                if use_amp:
                    with autocast(device_type='cuda'):
                        fake_clear = generator(haze)
                        fake_pred = discriminator(haze[:, :3], fake_clear)
                        g_gan_loss = hinge_loss_g(fake_pred) * args.lambda_gan
                        g_l1_loss = l1_loss(fake_clear, clear) * args.lambda_l1
                        g_perceptual_loss = perceptual_loss(fake_clear, clear) * args.lambda_perceptual
                        g_loss = g_gan_loss + g_l1_loss + g_perceptual_loss
                else:
                    fake_clear = generator(haze)
                    fake_pred = discriminator(haze[:, :3], fake_clear)
                    g_gan_loss = hinge_loss_g(fake_pred) * args.lambda_gan
                    g_l1_loss = l1_loss(fake_clear, clear) * args.lambda_l1
                    g_perceptual_loss = perceptual_loss(fake_clear, clear) * args.lambda_perceptual
                    g_loss = g_gan_loss + g_l1_loss + g_perceptual_loss
                
                if use_amp:
                    scaler.scale(g_loss).backward()
                    scaler.unscale_(g_optimizer)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    scaler.step(g_optimizer)
                    scaler.update()
                else:
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    g_optimizer.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

                if (batch_idx + 1) % 50 == 0:
                    logging.info(f"  Batch [{batch_idx+1}/{len(train_loader)}] | D: {d_loss.item():.6f} | G: {g_loss.item():.6f}")

            avg_g_loss = epoch_g_loss / len(train_loader)
            avg_d_loss = epoch_d_loss / len(train_loader)

            # ========== 验证 ==========
            if (epoch + 1) % args.val_every == 0:
                generator.eval()
                val_l1 = 0.0
                val_psnr = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        haze = batch['haze'].to(device, non_blocking=True)
                        clear = batch['clear'].to(device, non_blocking=True)
                        fake = generator(haze)
                        l1 = l1_loss(fake, clear)
                        mse = F.mse_loss(fake, clear)
                        psnr = 10 * torch.log10(4.0 / mse)
                        val_l1 += l1.item()
                        val_psnr += psnr.item()
                
                avg_val_l1 = val_l1 / len(val_loader)
                avg_val_psnr = val_psnr / len(val_loader)

                logging.info(f"Epoch [{epoch+1}/{args.epochs}] | "
                             f"D Loss: {avg_d_loss:.6f} | "
                             f"G Loss: {avg_g_loss:.6f} | "
                             f"Val L1: {avg_val_l1:.6f} | "
                             f"Val PSNR: {avg_val_psnr:.2f} dB | "
                             f"LR_G: {g_optimizer.param_groups[0]['lr']:.2e}")

                # 基于验证损失更新调度器
                #  g_scheduler.step(avg_val_l1)
                #  d_scheduler.step(avg_val_l1)

                if avg_val_l1 < best_val_loss:
                    best_val_loss = avg_val_l1
                    torch.save(generator.state_dict(),
                               os.path.join(args.save_dir, 'best_generator.pth'))
                    logging.info(f"  -> Saved best generator (val_l1={best_val_loss:.6f})")
            else:
                logging.info(f"Epoch [{epoch+1}/{args.epochs}] | "
                             f"D Loss: {avg_d_loss:.6f} | "
                             f"G Loss: {avg_g_loss:.6f} | "
                             f"LR_G: {g_optimizer.param_groups[0]['lr']:.2e}")

            # 更新学习率调度器（每个 epoch 结束后调用一次）
            g_scheduler.step()
            d_scheduler.step()

            # ========== 保存 last checkpoint ==========
            checkpoint_state = {
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': g_scheduler.state_dict(),
                'd_scheduler_state_dict': d_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
            }
            save_checkpoint(checkpoint_state,
                            os.path.join(args.save_dir, 'last_checkpoint.pth'))

    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user. Saving checkpoint before exit...")
        checkpoint_state = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_scheduler_state_dict': g_scheduler.state_dict(),
            'd_scheduler_state_dict': d_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
        }
        save_checkpoint(checkpoint_state,
                        os.path.join(args.save_dir, 'last_checkpoint.pth'))
        sys.exit(0)

    logging.info("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN for image dehazing with depth guidance')
    
    parser.add_argument('--haze_dir', type=str, default='train/hazy')
    parser.add_argument('--depth_dir', type=str, default='train/depth')
    parser.add_argument('--clear_dir', type=str, default='train/clear')
    parser.add_argument('--save_dir', type=str, default='checkpoints_gan')
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--expand_factor', type=int, default=2)

    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    
    parser.add_argument('--lambda_gan', type=float, default=30.0)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--lambda_perceptual', type=float, default=10.0)
    
    parser.add_argument('--compile', action='store_true')
    
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=16)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no-use_augmentation', action='store_false')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file (default: <save_dir>/train.log)')
    # 半精度开关：默认启用，使用 --no-fp16 禁用
    parser.add_argument('--no-fp16', action='store_true', help='Disable mixed precision (default: enabled)')
    
    args = parser.parse_args()
    
    # 若未指定 log_file，默认保存到 save_dir/train.log
    log_file = args.log_file if args.log_file else os.path.join(args.save_dir, 'train.log')
    
    # 设置日志
    setup_logging(log_file=log_file)
    
    logging.info("=" * 60)
    logging.info("Training Configuration:")
    logging.info(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.info(f"  Mixed precision: {'enabled' if not args.no_fp16 and torch.cuda.is_available() else 'disabled'}")
    logging.info(f"  Dataset random seed: {args.random_seed}")
    logging.info(f"  Dataset expand factor: {args.expand_factor}")
    logging.info(f"  Use augmentation: {args.no_use_augmentation}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Learning rates: G={args.lr_g}, D={args.lr_d}")
    logging.info(f"  Loss weights: GAN={args.lambda_gan}, L1={args.lambda_l1}, Perceptual={args.lambda_perceptual}")
    logging.info(f"  Weight decay: {args.weight_decay}")
    logging.info(f"  Compile: {args.compile}")
    logging.info(f"  Log file: {log_file}")
    logging.info("=" * 60)
    
    train(args)
