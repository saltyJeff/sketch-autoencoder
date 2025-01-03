import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod, sqrt

class LayerNorm2d(nn.GroupNorm):
    def __init__(self, chans: int = 1, affine: bool = False):
        super().__init__(1, chans, affine=affine)

class ConvNextBlock(nn.Module):
    def __init__(self, chans: int, kernel_size: int=3, stride: bool = False, hidden_chan_factor: float=4):
        super().__init__()
        self.hidden_chans = int(chans * hidden_chan_factor)
        stride_size = kernel_size//2 + 1 if stride else 1
        self.block = nn.Sequential(
            nn.Conv2d(chans, chans, kernel_size=kernel_size, stride=stride_size, padding=kernel_size//2, bias=False),
            LayerNorm2d(),
            nn.Conv2d(chans, self.hidden_chans, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_chans, chans, kernel_size=1)
        )
    def forward(self, x: torch.Tensor):
        return x + self.block(x)

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