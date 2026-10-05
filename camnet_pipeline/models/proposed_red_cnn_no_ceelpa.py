import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, f):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(f, f//16, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(f//16, f, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(F.adaptive_avg_pool2d(x, 1)) + self.fc(F.adaptive_max_pool2d(x, 1)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class PixelAttention(nn.Module):
    def __init__(self, f):
        super(PixelAttention, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(f, f, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.conv(x)

class RCA_Block(nn.Module):
    def __init__(self, f):
        super(RCA_Block, self).__init__()
        self.ca = ChannelAttention(f)
        self.sa = SpatialAttention()
        self.pa = PixelAttention(f)
    def forward(self, x):
        return self.pa(x * self.ca(x) * self.sa(x))

class MSFE_Block(nn.Module):
    def __init__(self, f):
        super(MSFE_Block, self).__init__()
        self.conv1 = nn.Conv2d(f, f//3, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(f, f//3, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(f, f//3, 3, padding=3, dilation=3)
        self.fuse  = nn.Conv2d(f, f, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        return self.lrelu(self.fuse(out) + x)

class Proposed_RED_CNN_No_CEELPA(nn.Module):
    """
    Ablation Study: Proposed Model WITHOUT CEELPA
    - MSFE with Hybrid Parallel Dilation (1, 2, 3)
    - RCA (Spatial, Channel, Pixel)
    - Standard Convolution Decoders
    - Standard Additive Skip Connections (No Masks)
    """
    def __init__(self, in_channels=1, out_channels=1, filters=96):
        super(Proposed_RED_CNN_No_CEELPA, self).__init__()
        
        self.enc1 = nn.Conv2d(in_channels, filters, 5, padding=2)
        
        self.rca1, self.msfe1 = RCA_Block(filters), MSFE_Block(filters)
        self.rca2, self.msfe2 = RCA_Block(filters), MSFE_Block(filters)
        self.rca3, self.msfe3 = RCA_Block(filters), MSFE_Block(filters)
        
        self.enc_bottleneck = nn.Conv2d(filters, filters, 5, padding=2)
        
        self.dec6, self.dec7 = nn.Conv2d(filters, filters, 5, padding=2), nn.Conv2d(filters, filters, 5, padding=2)
        self.dec8, self.dec9 = nn.Conv2d(filters, filters, 5, padding=2), nn.Conv2d(filters, filters, 5, padding=2)
        self.dec10 = nn.ConvTranspose2d(filters, out_channels, 5, padding=2)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.lrelu(self.enc1(x))
        e2 = self.msfe1(self.rca1(e1))
        e3 = self.msfe2(self.rca2(e2))
        e4 = self.msfe3(self.rca3(e3))
        e5 = self.lrelu(self.enc_bottleneck(e4))
        
        # Standard Additive Skip Connections (No CEELPA Masks)
        d6 = self.lrelu(self.dec6(e5) + e4)
        d7 = self.lrelu(self.dec7(d6))
        d8 = self.lrelu(self.dec8(d7) + e2)
        d9 = self.lrelu(self.dec9(d8))
        return self.dec10(d9) + x
