import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetPlusPlus, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Nodes X_i,j
        # X_0,0
        self.conv0_0 = ConvBlock(in_channels, features[0])
        # X_1,0
        self.conv1_0 = ConvBlock(features[0], features[1])
        # X_2,0
        self.conv2_0 = ConvBlock(features[1], features[2])
        # X_3,0
        self.conv3_0 = ConvBlock(features[2], features[3])
        # X_4,0 (bottleneck)
        self.conv4_0 = ConvBlock(features[3], features[3] * 2) # Or features[4] if defined

        # X_0,1
        self.conv0_1 = ConvBlock(features[0] + features[1], features[0])
        # X_1,1
        self.conv1_1 = ConvBlock(features[1] + features[2], features[1])
        # X_2,1
        self.conv2_1 = ConvBlock(features[2] + features[3], features[2])
        # X_3,1
        self.conv3_1 = ConvBlock(features[3]*2 + features[3], features[3]) # features[3]*2 from conv4_0 upsample

        # X_0,2
        self.conv0_2 = ConvBlock(features[0]*2 + features[1], features[0]) # X_0,0, X_0,1, up(X_1,1)
        # X_1,2
        self.conv1_2 = ConvBlock(features[1]*2 + features[2], features[1]) # X_1,0, X_1,1, up(X_2,1)
        # X_2,2
        self.conv2_2 = ConvBlock(features[2]*2 + features[3], features[2]) # X_2,0, X_2,1, up(X_3,1)

        # X_0,3
        self.conv0_3 = ConvBlock(features[0]*3 + features[1], features[0]) # X_0,0, X_0,1, X_0,2, up(X_1,2)
        # X_1,3
        self.conv1_3 = ConvBlock(features[1]*3 + features[2], features[1]) # X_1,0, X_1,1, X_1,2, up(X_2,2)

        # X_0,4
        self.conv0_4 = ConvBlock(features[0]*4 + features[1], features[0]) # X_0,0, X_0,1, X_0,2, X_0,3, up(X_1,3)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Column 0
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        
        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # Column 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final_conv(x0_4)
        return output

if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlus(in_channels=3, out_channels=1).to(device)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = model(dummy_input)
    print("Output shape:", output.shape)

    # Check parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}") 