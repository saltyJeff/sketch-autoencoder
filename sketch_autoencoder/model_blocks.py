import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod

@torch.jit.script
def instance_norm_2d(x: torch.Tensor, eps: float = 1e-4):
    var, mean = torch.var_mean(x, dim=(-1, -2), keepdim=True)
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

class ResBlock(nn.Module):
    def __init__(self, dims: int, kernel_size: int=3, stride: bool = False, hidden_dim_factor: float=2):
        super().__init__()
        self.hidden_dims = int(dims * hidden_dim_factor)
        stride_dims = kernel_size//2 + 1 if stride else 1
        self.block = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=kernel_size, stride=stride_dims, padding=kernel_size//2, bias=False),
            InstanceNorm2d(),
            nn.Conv2d(dims, self.hidden_dims, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dims, dims, kernel_size=1)
        )
    def forward(self, x: torch.Tensor):
        return x + self.block(x)

def inv_softplus(x: torch.Tensor, eps: float=1e-4) -> torch.Tensor:
    return x.expm1().clamp_min(eps).log()
def inv_linear(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return F.linear(x-linear.bias, linear.weight.T)

class HalfSoftplusLinear(nn.Module):
    def __init__(self, hidden_dims: int):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.layer = nn.Linear(self.hidden_dims, self.hidden_dims)
    def forward(self, x: torch.Tensor):
        x = self.layer(x)
        x0, x1 = torch.tensor_split(x, 2, dim=1)
        x1 = F.softplus(x1)
        x = torch.cat((x0, x1), dim=1)
        return x
    def invert(self, x: torch.Tensor):
        x0, x1 = torch.tensor_split(x, 2, dim=1)
        x1 = inv_softplus(x1)
        x = torch.cat((x0, x1), dim=1)
        x = inv_linear(x, self.layer)
        return x

class ImgEmbedder(nn.Module):
    def __init__(self, img_size: torch.Size, embed_dims: int, hidden_dims: int):
        super().__init__()
        self.img_size = img_size
        self.img_dims = prod(img_size)
        self.embed_dims = embed_dims

        self.conv = nn.Conv2d(self.img_size[0], self.img_size[0], kernel_size=3, stride=2, padding=1)
        self.in_layer = nn.Linear(self.img_dims, hidden_dims)
        self.hidden = HalfSoftplusLinear(hidden_dims)
        self.out_layer = nn.Linear(hidden_dims, embed_dims)

    def img_to_embed(self, z: torch.Tensor) -> torch.Tensor:
        e = z.flatten(1)
        e = self.in_layer(e)
        e = self.hidden(e)
        e = self.out_layer(e)
        return e
    @torch.no_grad()
    def embed_to_img(self, e: torch.Tensor) -> torch.Tensor:
        e = inv_linear(e, self.out_layer)
        e = self.hidden.invert(e)
        e = inv_linear(e, self.in_layer)
        z = e.view(-1, *self.img_size)
        return z

class ImgChannelTransform(nn.Module):
    def __init__(self, img_size: torch.Size, output_chans: int, attn_chans: int):
        super().__init__()
        self.img_size = img_size
        self.img_dims = prod(img_size)
        self.pool_size = (img_size[0], img_size[1] // 2, img_size[2] // 2)
        self.pool_dims = prod(self.pool_size)
        self.hidden_size = (attn_chans, img_size[1], img_size[2])
        self.hidden_dims = prod(self.hidden_size)
        
        self.
        self.attn = nn.Linear(self.pool_dims, self.hidden_dims)
        self.block = nn.Sequential(
            nn.Conv2d(img_size[0] + attn_chans, img_size[0] + attn_chans, kernel_size=1),
            nn.SiLU(),
            InstanceNorm2d(),
        )
        self.out_layer = nn.Conv2d(img_size[0] + attn_chans, output_chans, kernel_size=1)

    def forward(self, x: torch.Tensor):
        a = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        a = a.flatten(1)
        a = self.attn(a)
        a = a.view(-1, *self.hidden_size)
        x = torch.cat((x, a), dim=1)
        x = x + self.block(x)
        x = self.out_layer(x)
        return x