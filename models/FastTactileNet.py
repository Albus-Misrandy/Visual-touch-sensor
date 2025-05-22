import torch
import torch.nn as nn
import torch.nn.functional as F


class FastTactileNet(nn.Module):
    def __init__(self):
        super(FastTactileNet, self).__init__()
        # 输入尺寸：3x480x640 (HxW)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 240x320
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True))

        # 倒残差块（MobileNetV2风格）
        self.block1 = InvertedResidual(16, 24, stride=2)  # 120x160
        self.block2 = InvertedResidual(24, 32, stride=2)  # 60x80
        self.block3 = InvertedResidual(32, 48, stride=2)  # 30x40

        # 注意力机制
        self.se = SqueezeExcitation(48)

        # 输出层
        self.final_conv = nn.Conv2d(48, 128, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.se(x)
        x = self.final_conv(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * 2)

        self.conv = nn.Sequential(
            # PW
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # DW
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # PW-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.shortcut = (stride == 1) and (inp == oup)

    def forward(self, x):
        if self.shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)