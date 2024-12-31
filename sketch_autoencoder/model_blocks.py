import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod

@torch.jit.script
def instance_norm_2d(x: torch.Tensor, eps: float = 1e-4):
    var, mean = torch.var_mean(x, dim=(-2, -1), keepdim=True)
    return (x - mean) / (torch.sqrt(var + eps))

class InstanceNorm2d(nn.Module):
    """Replacement for nn.InstanceNorm2D that respects channels_last"""
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return instance_norm_2d(x, self.eps)
    

class ScaleTanh(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
    def forward(self, x: torch.Tensor):
        return F.tanh(x / self.scale) * self.scale

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
    return F.linear(x, torch.linalg.pinv(linear.weight))

class ImgEmbedder(nn.Module):
    def __init__(self, embed_dim: int, img_size: torch.Size, hidden_dims: int):
        super().__init__()
        self.img_size = img_size
        self.img_dims = prod(self.img_size)
        self.eps = 1e-4

        self.z_to_h = nn.Linear(self.img_dims, hidden_dims)
        self.h_to_e = nn.Linear(hidden_dims, embed_dim, bias=False)

    def img_to_embed(self, z: torch.Tensor) -> torch.Tensor:
        z = z.flatten(1)
        h = self.z_to_h(z)
        e = self.h_to_e(h)
        return e
    
    @torch.no_grad()
    def embed_to_img(self, e: torch.Tensor) -> torch.Tensor:
        h = inv_linear(self.h_to_e, e)
        z = inv_linear(self.z_to_h, h)
        z = z.view(-1, *self.img_size)
        return z

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.img_to_embed(z)