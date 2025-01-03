import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod, sqrt

class LayerNorm2d(nn.GroupNorm):
    def __init__(self, chans: int = 1, affine: bool = False):
        super().__init__(1, chans, affine=affine)

class DownBlock(nn.Module):
    def __init__(self, chans: int):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(chans, 2*chans, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(),
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

class ReEncoder(nn.Module):
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
        z = F.linear(z, torch.linalg.pinv(self.weight))
        z = z.permute(0, 3, 1, 2) # NHWC -> NCHW
        return z