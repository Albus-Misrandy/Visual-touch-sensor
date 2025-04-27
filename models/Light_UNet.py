import torch
import torch.nn as nn


# 深度可分离卷积模块
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 编码器模块
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
        return x, p


# 解码器模块
class DecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_up, out_channels):
        """
        Args:
            in_channels_skip (int): 跳跃连接的通道数
            in_channels_up (int): 上采样后的通道数
            out_channels (int): 输出通道数
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            SeparableConv2d(in_channels_skip + in_channels_up, out_channels),  # 修正输入通道
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# 轻量化U-Net模型
class LightweightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_channels=32):
        super().__init__()
        # 编码器
        self.enc1 = EncoderBlock(in_channels, init_channels)
        self.enc2 = EncoderBlock(init_channels, init_channels * 2)
        self.enc3 = EncoderBlock(init_channels * 2, init_channels * 4)
        self.enc4 = EncoderBlock(init_channels * 4, init_channels * 8)

        # 中间层
        self.bottleneck = nn.Sequential(
            SeparableConv2d(init_channels * 8, init_channels * 16),
            nn.BatchNorm2d(init_channels * 16),
            nn.ReLU(inplace=True)
        )

        # 修正解码器（关键修改部分）
        self.dec4 = DecoderBlock(
            in_channels_skip=init_channels * 8,  # 对应enc4的输出通道
            in_channels_up=init_channels * 16,  # 对应bottleneck的输出通道
            out_channels=init_channels * 8
        )
        self.dec3 = DecoderBlock(
            in_channels_skip=init_channels * 4,  # 对应enc3的输出通道
            in_channels_up=init_channels * 8,  # 对应dec4的输出通道
            out_channels=init_channels * 4
        )
        self.dec2 = DecoderBlock(
            in_channels_skip=init_channels * 2,  # 对应enc2的输出通道
            in_channels_up=init_channels * 4,  # 对应dec3的输出通道
            out_channels=init_channels * 2
        )
        self.dec1 = DecoderBlock(
            in_channels_skip=init_channels,  # 对应enc1的输出通道
            in_channels_up=init_channels * 2,  # 对应dec2的输出通道
            out_channels=init_channels
        )
        # 输出层
        self.final = nn.Conv2d(init_channels, out_channels, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.final(d1)
