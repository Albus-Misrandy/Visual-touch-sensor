import torch
import torch.nn as nn
import torch.nn.functional as F


# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 = 深度卷积 + 逐点卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels  # 关键参数：每个输入通道独立卷积
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class StateAwareEncoder(nn.Module):
    """状态感知编码器"""

    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            DepthwiseSeparableConv(16, 32),
            DepthwiseSeparableConv(32, 64)
        )
        self.state_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        state_prob = self.state_head(features)
        return features, state_prob


class BenchmarkGenerator(nn.Module):
    """基准生成器"""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
        # 基准记忆体
        self.benchmark_prototype = nn.Parameter(torch.randn(1, 64, 16, 12))

    def forward(self, x):
        # 与基准原型对齐
        x = F.normalize(x, dim=1)
        bench = F.normalize(self.benchmark_prototype, dim=1)
        aligned = x + (bench - x) * 0.5  # 渐进式对齐
        return self.decoder(aligned)


class SABGNet(nn.Module):
    """完整网络"""

    def __init__(self):
        super().__init__()
        self.encoder = StateAwareEncoder()
        self.generator = BenchmarkGenerator()

    def forward(self, x):
        feat, prob = self.encoder(x)

        if self.training:
            # 训练时软融合
            bench_img = self.generator(feat)
            output = prob * bench_img + (1 - prob) * x
        else:
            # 推理时硬判决
            if prob > 0.5:
                output = self.generator(feat)
            else:
                output = x

        return output, prob
