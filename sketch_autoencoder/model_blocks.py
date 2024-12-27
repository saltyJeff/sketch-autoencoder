import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, dims: int, kernel_size: int=7, hidden_dim_factor: float=2):
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