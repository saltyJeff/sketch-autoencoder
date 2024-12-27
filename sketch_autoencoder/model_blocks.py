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

class ResBlock(nn.Module):
    def __init__(self, dims: int, kernel_size: int=3, hidden_dim_factor: float=2):
        super().__init__()
        self.hidden_dims = int(dims * hidden_dim_factor)
        self.block = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=kernel_size, padding=kernel_size//2),
            InstanceNorm2d(),
            nn.Conv2d(dims, self.hidden_dims, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dims, dims, kernel_size=1)
        )
    def forward(self, x: torch.Tensor):
        return x + self.block(x)

class AdaNormBlock(nn.Module):
    def __init__(self, dims: int, cond_dims: int, kernel_size: int = 3, hidden_dim_factor: float=2):
        super().__init__()
        self.dims = dims
        self.resblock = ResBlock(dims, kernel_size, hidden_dim_factor)
        self.cond = nn.Conv2d(cond_dims, 3*dims, kernel_size=1)
        self.norm = InstanceNorm2d()
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        residual = x
        cond = self.cond(c)
        cond_bias, cond_scale, cond_gate = cond.tensor_split(3, dim=1)

        x = self.norm(x)
        x = (x * (1 + cond_scale)) + cond_bias
        x = self.resblock(x)
        return residual + cond_gate * x

class ScaleTanh(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale
    def forward(self, x: torch.Tensor):
        return self.scale * F.tanh(x / self.scale)

@torch.jit.script
def inv_softplus(x: torch.Tensor, eps: float=1e-4) -> torch.Tensor:
    return x.expm1().clamp_min(eps).log()
def inv_linear(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return F.linear(x-linear.bias, linear.weight.T)

class ImgEmbedder(nn.Module):
    def __init__(self, img_size: torch.Size, embed_dims: int, num_hidden: int = 6, hidden_dim_factor: int = 8):
        super().__init__()
        self.img_size = img_size
        self.img_dims = prod(self.img_size)
        self.hidden_dims = embed_dims * hidden_dim_factor
        self.embed_dims = embed_dims

        self.in_layer = nn.Linear(self.img_dims, self.hidden_dims)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(self.hidden_dims, self.hidden_dims) for _ in range(num_hidden)
        )
        self.out_layer = nn.Linear(self.hidden_dims, self.embed_dims)

    def img_to_embed(self, z: torch.Tensor) -> torch.Tensor:
        z = z.flatten(1)
        z = self.in_layer(z)
        for layer in self.hidden_layers:
            z = layer(z)
            z = F.softplus(z)
        z = self.out_layer(z)
        return z
    @torch.no_grad()
    def embed_to_img(self, e: torch.Tensor) -> torch.Tensor:
        e = inv_linear(e, self.out_layer)
        for layer in self.hidden_layers:
            e = inv_softplus(e)
            e = inv_linear(e, layer)
        e = inv_linear(e, self.in_layer)
        z = e.view(-1, *self.img_size)
        return z