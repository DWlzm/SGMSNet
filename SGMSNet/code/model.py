"""
SAM (分割注意力网络) 实现
参考: "SAM: Spatial Attention and Memory for Medical Image Segmentation"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialAttentionModule(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 空间注意力权重
        attn = self.conv1(x)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        
        # 应用注意力
        out = x * attn
        return out, attn

class MemoryModule(nn.Module):
    """记忆模块，用于捕获长距离依赖"""
    def __init__(self, channels, num_memories=64):
        super(MemoryModule, self).__init__()
        self.num_memories = num_memories
        
        # 内存键和值
        self.memory_keys = nn.Parameter(torch.randn(1, num_memories, channels))
        self.memory_values = nn.Parameter(torch.randn(1, num_memories, channels))
        
        # 查询转换
        self.query_transform = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 输出转换
        self.output_transform = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 转换输入为查询
        queries = self.query_transform(x)  # B, C, H, W
        queries = queries.view(b, c, -1).permute(0, 2, 1)  # B, HW, C
        
        # 计算查询和内存键之间的相似度
        keys = self.memory_keys.expand(b, -1, -1)  # B, M, C
        attn_logits = torch.matmul(queries, keys.transpose(1, 2))  # B, HW, M
        attn_weights = F.softmax(attn_logits, dim=2)  # B, HW, M
        
        # 获取内存值
        values = self.memory_values.expand(b, -1, -1)  # B, M, C
        
        # 加权聚合内存值
        memory_output = torch.matmul(attn_weights, values)  # B, HW, C
        memory_output = memory_output.permute(0, 2, 1).view(b, c, h, w)  # B, C, H, W
        
        # 融合原始特征和内存输出
        output = x + memory_output
        output = self.output_transform(output)
        
        return output

class GatedAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(GatedAttention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: 主分支特征 (decoder)
        # x: skip特征 (encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DecoderBlock(nn.Module):
    """解码器块"""
    def __init__(self, in_channels, out_channels, use_attention=True, skip_channels=None):
        super(DecoderBlock, self).__init__()
        self.use_attention = use_attention
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if use_attention and skip_channels is not None:
            main_channels = in_channels - skip_channels
            self.attn = SpatialAttentionModule(out_channels)
            self.gate = GatedAttention(F_g=main_channels, F_l=skip_channels, F_int=out_channels // 2)
        elif use_attention:
            self.attn = SpatialAttentionModule(out_channels)
            self.gate = None
        else:
            self.gate = None
            self.attn = None
        
    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            if self.use_attention and self.skip_channels is not None and self.gate is not None:
                main_channels = self.in_channels - self.skip_channels
                main = x[:, :main_channels, ...] if x.shape[1] > self.skip_channels else x
                skip = self.gate(main, skip)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        if self.use_attention and self.attn is not None:
            x, _ = self.attn(x)
        
        return x

class SAM(nn.Module):
    """分割注意力网络模型"""
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, use_memory=True):
        super(SAM, self).__init__()
        self.use_memory = use_memory
        
        # 使用ResNet-34作为编码器
        self.encoder = models.resnet34(pretrained=True)
        
        # 适应输入通道数
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 编码器各阶段
        self.enc1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)  # 输出尺寸: 1/2, 64
        self.enc2 = nn.Sequential(self.encoder.maxpool, self.encoder.layer1)  # 输出尺寸: 1/4, 64
        self.enc3 = self.encoder.layer2  # 输出尺寸: 1/8, 128
        self.enc4 = self.encoder.layer3  # 输出尺寸: 1/16, 256
        self.enc5 = self.encoder.layer4  # 输出尺寸: 1/32, 512
        
        # 中间层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.dec5 = DecoderBlock(512 + 512, 256, use_attention=True, skip_channels=512)
        self.dec4 = DecoderBlock(256 + 256, 128, use_attention=True, skip_channels=256)
        self.dec3 = DecoderBlock(128 + 128, 64, use_attention=True, skip_channels=128)
        self.dec2 = DecoderBlock(64 + 64, 64, use_attention=False, skip_channels=64)
        self.dec1 = DecoderBlock(64, 64, use_attention=False, skip_channels=None)
        
        # 输出层
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # 记录原始输入尺寸
        input_size = x.size()
        
        # 编码器
        enc1 = self.enc1(x)  # 1/2, 64
        enc2 = self.enc2(enc1)  # 1/4, 64
        enc3 = self.enc3(enc2)  # 1/8, 128
        enc4 = self.enc4(enc3)  # 1/16, 256
        enc5 = self.enc5(enc4)  # 1/32, 512
        
        # 瓶颈
        bottleneck = self.bottleneck(enc5)
        
        # 解码器
        dec5 = self.dec5(bottleneck, enc5)
        dec4 = self.dec4(dec5, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2)
        
        # 输出层
        output = self.final(dec1)
        
        # 上采样到原始尺寸
        output = F.interpolate(output, size=(input_size[2], input_size[3]), mode='bilinear', align_corners=True)
        
        return output

class SAMLite(nn.Module):
    """轻量级分割注意力网络模型"""
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super(SAMLite, self).__init__()
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # 输出尺寸: 1/2, 32
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # 输出尺寸: 1/4, 64
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # 输出尺寸: 1/8, 128
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # 输出尺寸: 1/16, 256
        
        # 瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True)
        )
        
        # 空间注意力
        self.attention = SpatialAttentionModule(base_filters * 8)
        
        # 解码器
        self.dec4 = DecoderBlock(base_filters * 8 + base_filters * 8, base_filters * 4, use_attention=True, skip_channels=base_filters * 8)
        self.dec3 = DecoderBlock(base_filters * 4 + base_filters * 4, base_filters * 2, use_attention=True, skip_channels=base_filters * 4)
        self.dec2 = DecoderBlock(base_filters * 2 + base_filters * 2, base_filters, use_attention=False, skip_channels=base_filters * 2)
        self.dec1 = DecoderBlock(base_filters, base_filters, use_attention=False, skip_channels=None)
        
        # 输出层
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)
    
    def forward(self, x):
        # 记录原始输入尺寸
        input_size = x.size()
        
        # 编码器
        enc1 = self.enc1(x)  # 1/2, 32
        enc2 = self.enc2(enc1)  # 1/4, 64
        enc3 = self.enc3(enc2)  # 1/8, 128
        enc4 = self.enc4(enc3)  # 1/16, 256
        
        # 瓶颈
        bottleneck = self.bottleneck(enc4)
        
        # 空间注意力
        bottleneck, _ = self.attention(bottleneck)
        
        # 解码器
        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2)
        
        # 输出层
        output = self.final(dec1)
        
        # 上采样到原始尺寸
        output = F.interpolate(output, size=(input_size[2], input_size[3]), mode='bilinear', align_corners=True)
        
        return output

if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试SAM模型
    model = SAM(in_channels=3, out_channels=1).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    print(f"SAM 输入形状: {x.shape}")
    print(f"SAM 输出形状: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SAM 可训练参数量: {total_params:,}")
    
    # 测试SAMLite模型
    model_lite = SAMLite(in_channels=3, out_channels=1).to(device)
    output_lite = model_lite(x)
    print(f"\nSAMLite 输入形状: {x.shape}")
    print(f"SAMLite 输出形状: {output_lite.shape}")
    total_params_lite = sum(p.numel() for p in model_lite.parameters() if p.requires_grad)
    print(f"SAMLite 可训练参数量: {total_params_lite:,}") 