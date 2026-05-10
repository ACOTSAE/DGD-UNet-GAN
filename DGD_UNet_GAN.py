#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ========================= Basic Modules =========================

class ResBlock(nn.Module):
    """Residual block with GroupNorm (replaced InstanceNorm)"""
    def __init__(self, in_ch, out_ch, use_se=False, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.se = nn.Identity()   # SE blocks removed permanently
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        out = self.act1(out + identity)
        return out


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with reduced dilations [1,2] (fix for gridding)"""
    def __init__(self, in_ch, out_ch, dilations=[1, 2], num_groups=8):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.norms = nn.ModuleList([nn.GroupNorm(num_groups, out_ch) for _ in dilations])
        self.act = nn.GELU()
        self.fusion = nn.Conv2d(len(dilations) * out_ch, out_ch, 1)

    def forward(self, x):
        outs = []
        for conv, norm in zip(self.convs, self.norms):
            out = self.act(norm(conv(x)))
            outs.append(out)
        out = torch.cat(outs, dim=1)
        out = self.fusion(out)
        return out


# ========================= Simplified Swin Transformer Block (Fixed LayerNorm) =========================

class SwinBlock(nn.Module):
    """Simplified window self-attention + MLP (fixed missing layer norm)"""
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        Hp, Wp = x.shape[2], x.shape[3]

        x_w = x.reshape(B, C, Hp // self.window_size, self.window_size,
                         Wp // self.window_size, self.window_size)
        x_w = x_w.permute(0, 2, 4, 3, 5, 1).reshape(-1, self.window_size * self.window_size, C)

        # ** FIX: Apply LayerNorm before attention **
        x_w = self.norm1(x_w)
        x_w, _ = self.attn(x_w, x_w, x_w)

        x_w = x_w.reshape(B, Hp // self.window_size, Wp // self.window_size,
                          self.window_size, self.window_size, C)
        x_w = x_w.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            x_w = x_w[:, :, :H, :W]

        x_w = x + x_w
        x_flat = x_w.reshape(B, C, -1).transpose(1, 2)
        x_flat = self.norm2(x_flat)
        x_mlp = self.mlp(x_flat).transpose(1, 2).reshape(B, C, H, W)
        out = x_w + x_mlp
        return out


# ========================= Improved Downsampling (Strided Conv + GroupNorm) =========================

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.GELU()
        self.res = ResBlock(out_ch, out_ch, use_se=False, num_groups=num_groups)

    def forward(self, x):
        return self.res(self.act(self.norm(self.conv(x))))


# ========================= Improved Upsampling (Bilinear + 3x3 Conv for anti-aliasing) =========================

class Up(nn.Module):
    def __init__(self, up_in_ch, up_out_ch, skip_ch, out_ch, num_groups=8):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(up_in_ch, up_out_ch, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.res = ResBlock(up_out_ch + skip_ch, out_ch, use_se=False, num_groups=num_groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2], mode='reflect')
        x = torch.cat([x2, x1], dim=1)
        return self.res(x)


# ========================= Gated Fusion (with gate clamping) =========================

class GatedFusion(nn.Module):
    def __init__(self, rgb_ch, depth_ch, out_ch):
        super().__init__()
        self.gate_conv = nn.Conv2d(rgb_ch + depth_ch, out_ch, 3, padding=1)
        nn.init.constant_(self.gate_conv.bias, 0.0)
        self.gate = nn.Sequential(
            self.gate_conv,
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(rgb_ch + depth_ch, out_ch, 1)

    def forward(self, rgb, depth):
        cat = torch.cat([rgb, depth], dim=1)
        gate = self.gate(cat)
        gate = torch.clamp(gate, 0.1, 0.9)
        out = self.proj(cat) * gate
        return out


# ========================= Depth Encoder (GroupNorm) =========================

class DepthEncoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=8, num_groups=8):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.GroupNorm(num_groups, base_filters),
            nn.GELU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, base_filters * 2),
            nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, base_filters * 4),
            nn.GELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, base_filters * 8),
            nn.GELU()
        )

    def forward(self, depth):
        f0 = self.conv0(depth)               # [B,8,H,W]
        f1 = self.down1(f0)                  # [B,16,H/2,W/2]
        f2 = self.down2(f1)                  # [B,32,H/4,W/4]
        f3 = self.down3(f2)                  # [B,64,H/8,W/8]
        return [f0, f1, f2, f3]


# ========================= Output Convolutions =========================

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class ColorCorrection(nn.Module):
    """Color correction with GroupNorm (replaced InstanceNorm)"""
    def __init__(self, in_ch=16, out_ch=3, num_groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.GELU(),
            nn.Conv2d(32, out_ch, 1)
        )

    def forward(self, x):
        return self.conv(x)


# ========================= Generator Main Architecture =========================

class DGD_UNet_GAN(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()

        # RGB encoder
        self.rgb_inc = ResBlock(3, 32, use_se=False, num_groups=8)
        self.rgb_down1 = Down(32, 64, num_groups=8)
        self.rgb_down2 = Down(64, 128, num_groups=8)
        self.rgb_down3 = Down(128, 256, num_groups=8)
        self.rgb_down4 = Down(256, 256, num_groups=8)

        # Depth encoder
        self.depth_enc = DepthEncoder(in_channels=1, base_filters=8, num_groups=8)

        # Bottleneck: ASPP (dilations [1,2]) + SwinBlock
        self.aspp = ASPP(256, 256, dilations=[1,2], num_groups=8)
        self.swin_bottleneck = SwinBlock(256, window_size=8)

        # Gated fusion
        self.fusion3 = GatedFusion(256, 64, 256)
        self.fusion2 = GatedFusion(128, 32, 128)
        self.fusion1 = GatedFusion(64, 16, 64)
        self.fusion0 = GatedFusion(32, 8, 32)

        # Decoder
        self.up1 = Up(256, 128, skip_ch=256, out_ch=128, num_groups=8)
        self.up2 = Up(128, 64, skip_ch=128, out_ch=64, num_groups=8)
        self.up3 = Up(64, 32, skip_ch=64, out_ch=32, num_groups=8)
        self.up4 = Up(32, 16, skip_ch=32, out_ch=16, num_groups=8)

        # Output layers with smoothing before final color correction
        self.outc = OutConv(16, 16)
        self.smooth = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),  # depthwise conv
            nn.Conv2d(16, 16, 1, bias=False)                        # pointwise conv
        )
        self.color_corr = ColorCorrection(16, out_channels, num_groups=8)

        # Learnable residual fusion weight (initial 0.5)
        self.rgb_residual_weight = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))

    def forward(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :]

        # RGB encode
        e0 = self.rgb_inc(rgb)          # 32
        e1 = self.rgb_down1(e0)         # 64
        e2 = self.rgb_down2(e1)         # 128
        e3 = self.rgb_down3(e2)         # 256
        e4 = self.rgb_down4(e3)         # 256

        # Depth encode
        depth_feats = self.depth_enc(depth)

        # Bottleneck
        b = self.aspp(e4)
        b = self.swin_bottleneck(b)

        # Decode with gated fusion
        d1 = self.up1(b, self.fusion3(e3, depth_feats[3]))
        d2 = self.up2(d1, self.fusion2(e2, depth_feats[2]))
        d3 = self.up3(d2, self.fusion1(e1, depth_feats[1]))
        d4 = self.up4(d3, self.fusion0(e0, depth_feats[0]))

        # Output smoothing and color correction
        feat = self.outc(d4)                         # [B,16,H,W]
        feat = self.smooth(feat)                     # suppress isolated spikes
        out = self.color_corr(feat)                  # [B,3,H,W]

        # Learnable residual fusion
        w = torch.sigmoid(self.rgb_residual_weight).view(1, 3, 1, 1)
        out = out * w + rgb * (1 - w)

        return out


# ========================= Discriminator (GroupNorm + Spectral Norm) =========================

class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, num_groups=8):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride, padding=1, bias=False))
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.LeakyReLU(0.2)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)),
                nn.GroupNorm(num_groups, out_ch)
            )

    def forward(self, x):
        # ** FIX: Activation after adding shortcut **
        out = self.act(self.norm(self.conv(x)) + self.shortcut(x))
        return out


class Discriminator(nn.Module):
    """70x70 PatchGAN discriminator"""
    def __init__(self, haze_channels=3, target_channels=3):
        super().__init__()
        in_channels = haze_channels + target_channels
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            DiscBlock(64, 128, stride=2, num_groups=8),
            DiscBlock(128, 256, stride=2, num_groups=8),
            spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, haze, target):
        x = torch.cat([haze, target], dim=1)
        return self.model(x)


# ========================= Generator Wrapper =========================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DGD_UNet_GAN()

    def forward(self, x):
        return self.model(x)


# ========================= Test Code =========================

if __name__ == '__main__':
    G = Generator()
    D = Discriminator()
    x = torch.randn(1, 4, 256, 256)
    with torch.no_grad():
        fake = G(x)
        score = D(fake, x[:, :3, :, :])
    print(f"Generator output shape: {fake.shape}")
    print(f"Discriminator output shape: {score.shape}")
    total_params_g = sum(p.numel() for p in G.parameters() if p.requires_grad)
    total_params_d = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"Generator params: {total_params_g / 1e6:.2f}M")
    print(f"Discriminator params: {total_params_d / 1e6:.2f}M")
    print("✅ All modules pass test.")
