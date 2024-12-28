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

class ImgEmbedder(nn.Module):
    def __init__(self, img_size: torch.Size, embed_dims: int, hidden_dims: int):
        super().__init__()
        self.img_dims = prod(img_size)
        self.embed_dims = embed_dims
        
        self.residual = nn.Linear(self.img_dims, hidden_dims // 2)
        self.features = nn.Sequential(
            nn.Linear(self.img_dims, hidden_dims // 2),
            nn.Softplus()
        )
        self.fc = nn.Linear(hidden_dims, embed_dims)

    def img_to_embed(self, z: torch.Tensor) -> torch.Tensor:
        e = z.flatten(1)
        res = self.residual(e)
        feat = self.features(e)
        e = torch.cat((res, feat), dim=1)
        e = self.fc(e)
        return e
    # @torch.no_grad()
    # def embed_to_img(self, e: torch.Tensor) -> torch.Tensor:
    #     e = inv_linear(e, self.out_layer)
    #     for layer in self.hidden_layers:
    #         e = inv_softplus(e)
    #         e = inv_linear(e, layer)
    #     e = inv_linear(e, self.in_layer)
    #     z = e.view(-1, *self.img_size)
    #     return z