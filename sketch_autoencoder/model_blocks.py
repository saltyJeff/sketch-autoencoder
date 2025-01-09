import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod, sqrt

class LayerNorm2d(nn.GroupNorm):
    def __init__(self, chans: int = -1):
        if chans < 0:
            chans = 1
            affine = False
        else:
            affine = True
        super().__init__(1, chans, affine=affine)

class DownBlock(nn.Module):
    def __init__(self, chans: int):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(chans, 2*chans, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(2*chans),
        )
        self.block = nn.Sequential(
            nn.Conv2d(2*chans, 4*chans, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4*chans, 2*chans, kernel_size=1)
        )
    def forward(self, x: torch.Tensor):
        x = self.in_conv(x)
        x = x + self.block(x)
        return x

class ImgEmbedder(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, first_chans: int, num_blocks: int):
        super().__init__()
        self.in_conv = nn.Conv2d(in_chans, first_chans, kernel_size=1)
        self.blocks = nn.Sequential(
            *[DownBlock(first_chans * 2**i) for i in range(num_blocks)]
        )
        last_chans = first_chans * (2**num_blocks)
        self.embedder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            LayerNorm2d(last_chans),
            nn.Flatten(),
            nn.Linear(last_chans, out_chans),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.blocks(x)
        x = self.embedder(x)
        return x
        
class InvertibleLinear(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dims, dims))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
    def forward(self, z: torch.Tensor):
        return self.encode(z)
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.permute(0, 2, 3, 1) # NCHW -> NHWC
        z = F.linear(z, self.weight)
        z = z.permute(0, 3, 1, 2) # NHWC -> NCHW
        return z
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.permute(0, 2, 3, 1) # NCHW -> NHWC
        z = F.linear(z, self.weight.T)
        z = z.permute(0, 3, 1, 2) # NHWC -> NCHW
        return z