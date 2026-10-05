import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        # Layer 1: Conv + ReLU
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Layers 2 to Depth-1: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, 
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # Layer Depth: Conv
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # The model predicts the NOISE (residual)
        # Final denoised image = Input - Residual
        out = self.dncnn(x)
        return x - out
