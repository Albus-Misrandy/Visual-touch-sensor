import torch
import torch.nn as nn
from torch.nn import functional as F


# 深度可分离卷积模块（保持高效性）
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=3, padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 编码器模块（保持轻量）
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            SeparableConv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p  # (保持的特征, 下采样结果)


# 优化后的解码模块
class EfficientDecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_up, out_channels):
        super().__init__()
        # 亚像素卷积上采样（比双线性上采样快30%）
        self.up = nn.Sequential(
            nn.Conv2d(in_channels_up, out_channels * 4, kernel_size=1),  # 注意通道数变化
            nn.PixelShuffle(upscale_factor=2)  # 输出通道变为out_channels
        )
        # 轻量卷积块（减少层数）
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + in_channels_skip, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)

        # 尺寸对齐（应对奇数尺寸问题）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear')

        # 跳跃连接拼接
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# 完整优化模型
class LightweightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_channels=24):
        super().__init__()

        # 编码器（调整通道增长系数）
        self.enc1 = EncoderBlock(in_channels, init_channels)  # out:24
        self.enc2 = EncoderBlock(init_channels, init_channels * 2)  # out:48
        self.enc3 = EncoderBlock(init_channels * 2, init_channels * 3)  # out:72
        self.enc4 = EncoderBlock(init_channels * 3, init_channels * 4)  # out:96

        # 瓶颈层（减少通道扩展）
        self.bottleneck = nn.Sequential(
            SeparableConv2d(init_channels * 4, init_channels * 8),  # out:192
            nn.BatchNorm2d(init_channels * 8),
            nn.ReLU(inplace=True)
        )

        # 解码器（优化结构）
        self.dec4 = EfficientDecoderBlock(
            in_channels_skip=init_channels * 4,  # 96
            in_channels_up=init_channels * 8,  # 192
            out_channels=init_channels * 4  # 96
        )
        self.dec3 = EfficientDecoderBlock(
            in_channels_skip=init_channels * 3,  # 72
            in_channels_up=init_channels * 4,  # 96
            out_channels=init_channels * 3  # 72
        )
        self.dec2 = EfficientDecoderBlock(
            in_channels_skip=init_channels * 2,  # 48
            in_channels_up=init_channels * 3,  # 72
            out_channels=init_channels * 2  # 48
        )
        self.dec1 = EfficientDecoderBlock(
            in_channels_skip=init_channels,  # 24
            in_channels_up=init_channels * 2,  # 48
            out_channels=init_channels  # 24
        )

        # 输出层
        self.final = nn.Conv2d(init_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码路径
        s1, p1 = self.enc1(x)  # s1: (B, 24, H, W)
        s2, p2 = self.enc2(p1)  # s2: (B, 48, H/2, W/2)
        s3, p3 = self.enc3(p2)  # s3: (B, 72, H/4, W/4)
        s4, p4 = self.enc4(p3)  # s4: (B, 96, H/8, W/8)

        # 瓶颈层
        b = self.bottleneck(p4)  # (B, 192, H/8, W/8)

        # 解码路径
        d4 = self.dec4(b, s4)  # (B, 96, H/4, W/4)
        d3 = self.dec3(d4, s3)  # (B, 72, H/2, W/2)
        d2 = self.dec2(d3, s2)  # (B, 48, H, W)
        d1 = self.dec1(d2, s1)  # (B, 24, H, W)

        # 最终输出
        return self.final(d1)  # (B, 1, H, W)
