import torch.nn as nn


class LiteTactileNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入尺寸：3x480x640
        self.encoder = nn.Sequential(
            # Stage 1 [3,480,640] -> [16,240,320]
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Stage 2 [16,240,320] -> [32,120,160]
            DepthwiseSeparableConv(16, 32, stride=2),

            # Stage 3 [32,120,160] -> [64,60,80]
            DepthwiseSeparableConv(32, 64, stride=2),

            # 保留空间信息
            nn.Conv2d(64, 128, 3, padding=1),  # [128,60,80]
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 动态阈值分支
        self.threshold_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [128,1,1]
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 重建解码器
        self.reconstruction = nn.Sequential(
            # [128,60,80] -> [64,120,160]
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),

            # [64,120,160] -> [32,240,320]
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),

            # [32,240,320] -> [16,480,640]
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1),

            # 最终输出
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)  # [B,128,60,80]
        threshold = self.threshold_layer(features)  # [B,1]
        recon = self.reconstruction(features)  # [B,3,480,640]
        return recon, threshold


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积(参数量减少75%)"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3,
                                   stride=stride, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))
