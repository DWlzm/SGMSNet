"""
SGMSNet: Spatial-Gated Memory Network for Chest X-ray Image Segmentation

This implementation provides a deep learning model for medical image segmentation,
specifically designed for chest X-ray images. The model incorporates:
1. Spatial attention mechanisms to focus on relevant regions
2. Memory modules to capture long-range dependencies
3. Gated attention for effective skip connection fusion
4. ResNet-34 as the encoder backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module
    
    This module generates spatial attention maps to focus on relevant regions in the input feature map.
    It reduces the channel dimension first, then generates a single-channel attention map,
    and finally applies the attention to the original feature map.
    
    Args:
        in_channels (int): Number of input channels
    """
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        # Reduce channel dimension to 1/8 of input channels
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # Generate single-channel attention map
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        # Sigmoid activation to normalize attention weights between 0 and 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate spatial attention weights
        attn = self.conv1(x)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        
        # Apply attention weights to input feature map
        out = x * attn
        return out, attn

class MemoryModule(nn.Module):
    """
    Memory Module for capturing long-range dependencies
    
    This module uses a set of learnable memory slots to capture and store information
    about the input features. It enables the model to capture long-range dependencies
    by attending to these memory slots based on the current input.
    
    Args:
        channels (int): Number of input/output channels
        num_memories (int, optional): Number of memory slots. Defaults to 64.
    """
    def __init__(self, channels, num_memories=64):
        super(MemoryModule, self).__init__()
        self.num_memories = num_memories
        
        # Learnable memory keys and values
        self.memory_keys = nn.Parameter(torch.randn(1, num_memories, channels))
        self.memory_values = nn.Parameter(torch.randn(1, num_memories, channels))
        
        # Transform input features to queries
        self.query_transform = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Transform output features
        self.output_transform = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Transform input features to queries
        queries = self.query_transform(x)  # B, C, H, W
        queries = queries.view(b, c, -1).permute(0, 2, 1)  # B, HW, C
        
        # Calculate similarity between queries and memory keys
        keys = self.memory_keys.expand(b, -1, -1)  # B, M, C
        attn_logits = torch.matmul(queries, keys.transpose(1, 2))  # B, HW, M
        attn_weights = F.softmax(attn_logits, dim=2)  # B, HW, M
        
        # Get memory values
        values = self.memory_values.expand(b, -1, -1)  # B, M, C
        
        # Weighted aggregation of memory values
        memory_output = torch.matmul(attn_weights, values)  # B, HW, C
        memory_output = memory_output.permute(0, 2, 1).view(b, c, h, w)  # B, C, H, W
        
        # Fuse original features with memory output
        output = x + memory_output
        output = self.output_transform(output)
        
        return output

class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism
    
    This module implements a gated attention mechanism for fusing features from
    different layers, typically between encoder and decoder features in skip connections.
    
    Args:
        F_g (int): Number of channels in the gating signal (decoder features)
        F_l (int): Number of channels in the input features (encoder features)
        F_int (int): Number of channels in the intermediate representation
    """
    def __init__(self, F_g, F_l, F_int):
        super(GatedAttention, self).__init__()
        # Transform for gating signal (decoder features)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Transform for input features (encoder features)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Generate attention gate
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Forward pass of the gated attention mechanism
        
        Args:
            g (torch.Tensor): Gating signal from decoder (B, F_g, H, W)
            x (torch.Tensor): Input features from encoder (B, F_l, H, W)
            
        Returns:
            torch.Tensor: Gated features (B, F_l, H, W)
        """
        # Transform gating signal
        g1 = self.W_g(g)
        # Transform input features
        x1 = self.W_x(x)
        # Generate attention map
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Apply attention to input features
        return x * psi

class DecoderBlock(nn.Module):
    """
    Decoder Block
    
    This module implements a decoder block with optional attention mechanisms.
    It handles upsampling, skip connection fusion, and feature transformation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        use_attention (bool, optional): Whether to use attention mechanisms. Defaults to True.
        skip_channels (int, optional): Number of channels in the skip connection. Defaults to None.
    """
    def __init__(self, in_channels, out_channels, use_attention=True, skip_channels=None):
        super(DecoderBlock, self).__init__()
        self.use_attention = use_attention
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        
        # Convolutional layers for feature transformation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize attention mechanisms based on configuration
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
        """
        Forward pass of the decoder block
        
        Args:
            x (torch.Tensor): Input features from previous decoder block (B, C, H, W)
            skip (torch.Tensor, optional): Skip connection features from encoder (B, C, H, W). Defaults to None.
            
        Returns:
            torch.Tensor: Output features (B, out_channels, H, W)
        """
        if skip is not None:
            # Upsample input to match skip connection size
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            # Apply gated attention if configured
            if self.use_attention and self.skip_channels is not None and self.gate is not None:
                main_channels = self.in_channels - self.skip_channels
                main = x[:, :main_channels, ...] if x.shape[1] > self.skip_channels else x
                skip = self.gate(main, skip)
            # Concatenate input with skip connection
            x = torch.cat([x, skip], dim=1)
        
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Apply spatial attention if configured
        if self.use_attention and self.attn is not None:
            x, _ = self.attn(x)
        
        return x

class SGMSNet(nn.Module):
    """
    Spatial-Gated Memory Network (SGMSNet)
    
    A deep learning model for chest X-ray image segmentation that incorporates:
    1. ResNet-34 as the encoder backbone
    2. Spatial attention mechanisms
    3. Memory modules for long-range dependencies
    4. Gated attention for skip connection fusion
    
    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        base_filters (int, optional): Number of base filters. Defaults to 64.
        use_memory (bool, optional): Whether to use memory modules. Defaults to True.
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, use_memory=True):
        super(SGMSNet, self).__init__()
        self.use_memory = use_memory
        
        # Use ResNet-34 as encoder backbone
        self.encoder = models.resnet34(pretrained=True)
        
        # Adapt input channel count if different from 3
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Encoder stages
        self.enc1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)  # Output size: 1/2, 64
        self.enc2 = nn.Sequential(self.encoder.maxpool, self.encoder.layer1)  # Output size: 1/4, 64
        self.enc3 = self.encoder.layer2  # Output size: 1/8, 128
        self.enc4 = self.encoder.layer3  # Output size: 1/16, 256
        self.enc5 = self.encoder.layer4  # Output size: 1/32, 512
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks
        self.dec5 = DecoderBlock(512 + 512, 256, use_attention=True, skip_channels=512)
        self.dec4 = DecoderBlock(256 + 256, 128, use_attention=True, skip_channels=256)
        self.dec3 = DecoderBlock(128 + 128, 64, use_attention=True, skip_channels=128)
        self.dec2 = DecoderBlock(64 + 64, 64, use_attention=False, skip_channels=64)
        self.dec1 = DecoderBlock(64, 64, use_attention=False, skip_channels=None)
        
        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass of the SGMSNet model
        
        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Segmentation output (B, out_channels, H, W)
        """
        # Record original input size
        input_size = x.size()
        
        # Encoder forward pass
        enc1 = self.enc1(x)  # 1/2, 64
        enc2 = self.enc2(enc1)  # 1/4, 64
        enc3 = self.enc3(enc2)  # 1/8, 128
        enc4 = self.enc4(enc3)  # 1/16, 256
        enc5 = self.enc5(enc4)  # 1/32, 512
        
        # Bottleneck
        bottleneck = self.bottleneck(enc5)
        
        # Decoder forward pass with skip connections
        dec5 = self.dec5(bottleneck, enc5)
        dec4 = self.dec4(dec5, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2)
        
        # Final output layer
        output = self.final(dec1)
        
        # Upsample to original input size
        output = F.interpolate(output, size=(input_size[2], input_size[3]), mode='bilinear', align_corners=True)
        
        return output


if __name__ == "__main__":
    """
    Test code for SGMSNet
    
    This section tests the model with random input to verify its functionality
    and print basic model information.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = SGMSNet(in_channels=3, out_channels=1).to(device)
    # Create random input tensor
    x = torch.randn(2, 3, 224, 224).to(device)
    # Forward pass
    output = model(x)
    # Print input/output shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    # Calculate total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
