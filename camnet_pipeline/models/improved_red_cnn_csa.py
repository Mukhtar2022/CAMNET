"""
improved_red_cnn_csa.py
========================
Implements: "Improved RED-CNN with Channel-Spatial Attention and Hybrid Loss
for Low-Dose CT", Zhang & Mei, CAIBDA 2025.
DOI: 10.1109/CAIBDA65784.2025.11182906

Architecture changes over baseline RED-CNN (Chen et al. 2017):
  - 128 filters (vs 96), same-padding 5x5 kernels (spatial size preserved)
  - Channel-Spatial Attention (CSA) after encoder layers 2, 3, 5
  - Skip connections: E2->D2, E4->D4 (+ global residual input->output)
  - Training loss: L_total = 1.0*MSE + 1.0*SSIM_loss  (alpha=beta=1.0)
  - 60 epochs, Adam lr=1e-4, decay x0.1 at epoch 20
"""

import torch
import torch.nn as nn


class ChannelSpatialAttention(nn.Module):
    """
    CSA module (Section III-C of the paper).

    Channel attention:
        z_c = (1/HW) * sum_{i,j} F_{c,i,j}          -- global avg pool
        s   = sigmoid( W2 * relu(W1 * z) )            -- two FC, ratio r=8
        F'  = s_c * F                                  -- channel-wise scale

    Spatial attention (3 conv layers on F'):
        A   = sigmoid( H3( relu( H2( relu( H1(F') ) ) ) ) )   -- 1xHxW map

    Output: F' * A  (element-wise)
    """

    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()
        reduced = max(1, channels // reduction_ratio)

        # Channel branch: GAP -> FC -> ReLU -> FC -> Sigmoid
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

        # Spatial branch: 3 x Conv2d  (C -> C//2 -> C//4 -> 1)
        c2 = max(1, channels // 2)
        c4 = max(1, channels // 4)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, c2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Channel attention ────────────────────────────────────────
        z = x.mean(dim=[2, 3])                         # (B, C)
        s = self.channel_fc(z).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        x_prime = x * s                                # (B, C, H, W)

        # ── Spatial attention ────────────────────────────────────────
        A = self.spatial_conv(x_prime)                 # (B, 1, H, W)

        return x_prime * A                             # (B, C, H, W)


class ImprovedREDCNN_CSA(nn.Module):
    """
    Improved RED-CNN with Channel-Spatial Attention (CSA Module).

    Encoder (5 layers, same padding, 128 ch):
        E1 -> E2 -> CSA -> E3 -> CSA -> E4 -> E5 -> CSA

    Decoder (5 layers, same padding):
        D1 -> D2 (+E2 skip) -> D3 -> D4 (+E4 skip) -> D5 (+x global residual)

    Training:  L = 1.0 * MSE + 1.0 * (1 - SSIM)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 filters: int = 128, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2   # 2 for 5x5 -> same-size feature maps

        # ── Encoder ──────────────────────────────────────────────────
        self.enc1 = nn.Conv2d(in_channels, filters, kernel_size, padding=pad)
        self.enc2 = nn.Conv2d(filters,     filters, kernel_size, padding=pad)
        self.csa2 = ChannelSpatialAttention(filters)          # after enc2
        self.enc3 = nn.Conv2d(filters,     filters, kernel_size, padding=pad)
        self.csa3 = ChannelSpatialAttention(filters)          # after enc3
        self.enc4 = nn.Conv2d(filters,     filters, kernel_size, padding=pad)
        self.enc5 = nn.Conv2d(filters,     filters, kernel_size, padding=pad)
        self.csa5 = ChannelSpatialAttention(filters)          # after enc5

        # ── Decoder ──────────────────────────────────────────────────
        self.dec1 = nn.ConvTranspose2d(filters,      filters,      kernel_size, padding=pad)
        self.dec2 = nn.ConvTranspose2d(filters,      filters,      kernel_size, padding=pad)
        self.dec3 = nn.ConvTranspose2d(filters,      filters,      kernel_size, padding=pad)
        self.dec4 = nn.ConvTranspose2d(filters,      filters,      kernel_size, padding=pad)
        self.dec5 = nn.ConvTranspose2d(filters, out_channels,      kernel_size, padding=pad)

        self.relu = nn.ReLU(inplace=True)

        # Gaussian weight initialisation: N(0, 0.01) as specified in paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ──────────────────────────────────────────────────
        e1 = self.relu(self.enc1(x))
        e2 = self.csa2(self.relu(self.enc2(e1)))      # CSA after layer 2
        e3 = self.csa3(self.relu(self.enc3(e2)))      # CSA after layer 3
        e4 = self.relu(self.enc4(e3))
        e5 = self.csa5(self.relu(self.enc5(e4)))      # CSA after layer 5

        # ── Decoder ──────────────────────────────────────────────────
        d1 = self.relu(self.dec1(e5))
        d2 = self.relu(self.dec2(d1) + e2)            # skip R2 from enc2
        d3 = self.relu(self.dec3(d2))
        d4 = self.relu(self.dec4(d3) + e4)            # skip R4 from enc4
        out = self.relu(self.dec5(d4) + x)            # global residual

        return out
