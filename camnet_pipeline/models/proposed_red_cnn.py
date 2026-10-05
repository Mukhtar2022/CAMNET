import torch
import torch.nn as nn
import torch.nn.functional as F

class CEELPA_Module(nn.Module):
    def __init__(self):
        super(CEELPA_Module, self).__init__()
        # Register Sobel kernels as non-trainable buffers
        sobel_x = torch.tensor([[-1., 0., 1.],
                               [-2., 0., 2.],
                               [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                               [0., 0., 0.],
                               [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        # Non‑trainable scalar parameters
        self.gamma = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.weight_edge = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.weight_contrast = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    def forward(self, x):
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        edge_prior = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        mean_val = torch.mean(x, dim=(2, 3), keepdim=True)
        contrast_prior = torch.exp(-(x - mean_val)**2 / (F.softplus(self.gamma) + 1e-6))
        edge_mask = torch.sigmoid(edge_prior * self.weight_edge)
        contrast_mask = torch.sigmoid(contrast_prior * self.weight_contrast)
        return edge_mask, contrast_mask

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class PixelAttention(nn.Module):
    def __init__(self, in_planes):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv1(x))

class RCA_Block(nn.Module):
    def __init__(self, filters):
        super(RCA_Block, self).__init__()
        self.ca = ChannelAttention(filters)
        self.sa = SpatialAttention()
        self.pa = PixelAttention(filters)
    def forward(self, x):
        return x * self.ca(x) * self.sa(x) * self.pa(x)

class MSFE_Block(nn.Module):
    def __init__(self, filters):
        super(MSFE_Block, self).__init__()
        self.b1 = nn.Conv2d(filters, filters // 3, kernel_size=3, padding=1, dilation=1)
        self.b2 = nn.Conv2d(filters, filters // 3, kernel_size=3, padding=2, dilation=2)
        self.b3 = nn.Conv2d(filters, filters // 3, kernel_size=3, padding=3, dilation=3)
        self.fuse = nn.Conv2d(filters, filters, kernel_size=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.lrelu(self.fuse(out) + x)

class Proposed_RED_CNN(nn.Module):
    """
    Stand-alone Champion Architecture (Matches V2 exactly)
    - Multiplicative Edge-Contrast Gating
    - RCA Multiplicative Integration
    - MSFE Parallel Dilation
    """
    def __init__(self, in_channels=1, out_channels=1, filters=96):
        super(Proposed_RED_CNN, self).__init__()
        self.ceelpa = CEELPA_Module()
        self.enc1 = nn.Conv2d(in_channels, filters, 5, padding=2)
        self.rca1, self.msfe1 = RCA_Block(filters), MSFE_Block(filters)
        self.rca2, self.msfe2 = RCA_Block(filters), MSFE_Block(filters)
        self.rca3, self.msfe3 = RCA_Block(filters), MSFE_Block(filters)
        self.enc_bottleneck = nn.Conv2d(filters, filters, 5, padding=2)
        self.dec6 = nn.Conv2d(filters, filters, 5, padding=2)
        self.dec7 = nn.Conv2d(filters, filters, 5, padding=2)
        self.dec8 = nn.Conv2d(filters, filters, 5, padding=2)
        self.dec9 = nn.Conv2d(filters, filters, 5, padding=2)
        self.dec10 = nn.ConvTranspose2d(filters, out_channels, 5, padding=2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        edge_mask, contrast_mask = self.ceelpa(x)
        e1 = self.lrelu(self.enc1(x))
        e2 = self.msfe1(self.rca1(e1))
        e3 = self.msfe2(self.rca2(e2))
        e4 = self.msfe3(self.rca3(e3))
        e5 = self.lrelu(self.enc_bottleneck(e4))
        
        # Multiplicative Masking for High Structural Similarity
        d6 = self.lrelu(self.dec6(e5) + e4 + (e4 * edge_mask))
        d7 = self.lrelu(self.dec7(d6))
        d8 = self.lrelu(self.dec8(d7) + e2 + (e2 * contrast_mask))
        d9 = self.lrelu(self.dec9(d8))
        return self.dec10(d9) + x
