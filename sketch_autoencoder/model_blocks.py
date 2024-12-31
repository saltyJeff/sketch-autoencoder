import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod

class ScaleTanh(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
    def forward(self, x: torch.Tensor):
        return F.tanh(x / self.scale) * self.scale

class DownBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, kernel_size: int = 3):
        super().__init__()
        stride = kernel_size//2 + 1
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
    def forward(self, x: torch.Tensor):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, kernel_size: int = 3):
        super().__init__()
        scale = kernel_size//2 + 1
        self.upsample = nn.Upsample(scale_factor=scale)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=kernel_size//2)
    def forward(self, x: torch.Tensor):
        return self.conv(self.upsample(x))

class ConvNextBlock(nn.Module):
    def __init__(self, chans: int, kernel_size: int=3, stride: bool = False, hidden_chan_factor: float=4):
        super().__init__()
        self.hidden_chans = int(chans * hidden_chan_factor)
        stride_size = kernel_size//2 + 1 if stride else 1
        self.block = nn.Sequential(
            nn.Conv2d(chans, chans, kernel_size=kernel_size, stride=stride_size, padding=kernel_size//2, bias=False),
            nn.GroupNorm(1, chans),
            nn.Conv2d(chans, self.hidden_chans, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_chans, chans, kernel_size=1)
        )
    def forward(self, x: torch.Tensor):
        return x + self.block(x)

def inv_linear(linear: nn.Linear, x: torch.Tensor):
    if linear.bias is not None:
        x = x - linear.bias
    return F.linear(x, linear.weight.T)

class ImgUnembedder(nn.Module):
    def __init__(self, embed_dim: int, img_size: torch.Size, hidden_dims: int):
        super().__init__()
        self.img_size = img_size
        self.img_dims = prod(self.img_size)

        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims),
            nn.Linear(hidden_dims, self.img_dims),
            ScaleTanh(3)
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        e = self.layers(e)
        z = e.view(-1, *self.img_size)
        return z