import torch.nn as nn

class LiteTactileNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入尺寸：3x480x640 (HxW)
        self.encoder = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 240x320
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Stage 2
            DepthwiseSeparableConv(16, 32, stride=2),  # 120x160
            
            # Stage 3
            DepthwiseSeparableConv(32, 64, stride=2),  # 60x80
            
            # 注意力压缩
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 动态阈值分支
        self.threshold_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 自监督任务头
        self.reconstruction = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)  # [B, 64, 1, 1]
        recon = self.reconstruction(features.view(-1,64,1,1))  # 重建图像
        threshold = self.threshold_layer(features.view(-1,64))  # 动态阈值
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