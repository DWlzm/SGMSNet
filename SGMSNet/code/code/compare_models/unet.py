import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    UNet中的基础模块，包含两个卷积层，每个卷积层后接一个ReLU激活函数。
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    UNet网络结构。
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        # 编码器部分
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        self.encoder5 = DoubleConv(512, 1024)

        # 解码器部分
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(128, 64)

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器前向传播
        x1 = self.encoder1(x)
        x2 = self.encoder2(F.max_pool2d(x1, 2))
        x3 = self.encoder3(F.max_pool2d(x2, 2))
        x4 = self.encoder4(F.max_pool2d(x3, 2))
        x5 = self.encoder5(F.max_pool2d(x4, 2))

        # 解码器前向传播
        up1 = self.up1(x5)
        merge1 = torch.cat([up1, x4], dim=1)
        d1 = self.decoder1(merge1)

        up2 = self.up2(d1)
        merge2 = torch.cat([up2, x3], dim=1)
        d2 = self.decoder2(merge2)

        up3 = self.up3(d2)
        merge3 = torch.cat([up3, x2], dim=1)
        d3 = self.decoder3(merge3)

        up4 = self.up4(d3)
        merge4 = torch.cat([up4, x1], dim=1)
        d4 = self.decoder4(merge4)

        # 输出层
        out = self.out_conv(d4)
        return out
    
    
# 定义损失函数
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 展平预测和目标
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        # 计算交集
        intersection = (pred_flat * target_flat).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, pred, target):
        if isinstance(pred, list):  # 深监督
            loss = 0
            weights = [0.5, 0.75, 0.9, 1.0] if len(pred) == 4 else [1.0/len(pred)] * len(pred)
            
            for i, p in enumerate(pred):
                if p.shape[2:] != target.shape[2:]:
                    p = F.interpolate(p, size=target.shape[2:], mode='bilinear', align_corners=True)
                loss += weights[i] * (self.alpha * self.bce_loss(p, target) + self.beta * self.dice_loss(p, target))
            
            return loss
        else:
            return self.alpha * self.bce_loss(pred, target) + self.beta * self.dice_loss(pred, target)