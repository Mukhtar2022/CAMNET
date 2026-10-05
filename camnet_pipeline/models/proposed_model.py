import torch
import torch.nn as nn

class ProposedModel(nn.Module):
    """
    Residual Encoder-Decoder CNN for Low-Dose CT Denoising.
    Initially reverted to RED-CNN baseline to implement improvements 'bit-by-bit'.

    Architecture (Chen et al. 2017):
    - 10 layers: 5 Conv (encoder) + 5 Deconv (decoder)
    - 96 filters, 5x5 kernels, no padding
    - 2 shortcut / skip connections: conv2->deconv8, conv4->deconv6
    - 1 global residual: input -> deconv10
    - Activation: ReLU after every layer
    """
    def __init__(self, in_channels=1, out_channels=1, filters=96, kernel_size=5):
        super(ProposedModel, self).__init__()

        # ── Encoder (no padding → size reduces each layer) ─────────────────
        self.enc1 = nn.Conv2d(in_channels, filters, kernel_size, padding=0)
        self.enc2 = nn.Conv2d(filters,     filters, kernel_size, padding=0)
        self.enc3 = nn.Conv2d(filters,     filters, kernel_size, padding=0)
        self.enc4 = nn.Conv2d(filters,     filters, kernel_size, padding=0)
        self.enc5 = nn.Conv2d(filters,     filters, kernel_size, padding=0)

        # ── Decoder (transposed conv, no padding → size grows each layer) ──
        self.dec6  = nn.ConvTranspose2d(filters,      filters,      kernel_size, padding=0)
        self.dec7  = nn.ConvTranspose2d(filters,      filters,      kernel_size, padding=0)
        self.dec8  = nn.ConvTranspose2d(filters,      filters,      kernel_size, padding=0)
        self.dec9  = nn.ConvTranspose2d(filters,      filters,      kernel_size, padding=0)
        self.dec10 = nn.ConvTranspose2d(filters, out_channels,      kernel_size, padding=0)

        self.relu  = nn.ReLU(inplace=True)

        # Weight initialisation: Gaussian (0, 0.01) as per paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # ── Encoder ─────────────────────────────────────────────────────────
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))   # shortcut_deconv8
        e3 = self.relu(self.enc3(e2))
        e4 = self.relu(self.enc4(e3))   # shortcut_deconv6
        e5 = self.relu(self.enc5(e4))

        # ── Decoder ─────────────────────────────────────────────────────────
        d6  = self.relu(self.dec6(e5) + e4)  # skip from enc4
        d7  = self.relu(self.dec7(d6))
        d8  = self.relu(self.dec8(d7) + e2)  # skip from enc2
        d9  = self.relu(self.dec9(d8))
        d10 = self.relu(self.dec10(d9) + x)  # global residual (input)

        return d10
