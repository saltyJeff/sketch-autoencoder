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

def inv_linear(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return F.linear(x-linear.bias, torch.linalg.pinv(linear.weight))

def inv_conv2d(x: torch.Tensor, conv: torch.nn.Conv2d, img_size: torch.Size) -> torch.Tensor:
    x = x - conv.bias.view(1, -1, 1, 1)
    
    # Calculate output_padding
    N, C, H, W = x.shape
    H_out = (H - 1) * conv.stride[0] - 2 * conv.padding[0] + (conv.kernel_size[0] - 1) + 1
    W_out = (W - 1) * conv.stride[1] - 2 * conv.padding[1] + (conv.kernel_size[1] - 1) + 1
    
    # Calculate required output_padding
    output_padding = (
        min(max(img_size[1] - H_out, 0), conv.stride[0] - 1),
        min(max(img_size[2] - W_out, 0), conv.stride[1] - 1)
    )
    
    # Apply transposed convolution
    return F.conv_transpose2d(
        input=x,
        weight=conv.weight.data,
        bias=None,
        stride=conv.stride,
        padding=conv.padding,
        output_padding=output_padding,
        groups=conv.groups,
        dilation=conv.dilation
    )
    
class ImgEmbedder(nn.Module):
    def __init__(self, img_size: torch.Size, embed_dims: int, hidden_dims: int):
        super().__init__()
        self.img_size = img_size
        self.img_dims = prod(img_size)
        self.conv_size = (img_size[0]*2, img_size[1] // 2, img_size[2] // 2)
        self.conv_dims = prod(self.conv_size)
        self.embed_dims = embed_dims

        self.in_conv = nn.Conv2d(img_size[0], img_size[0] * 2, kernel_size=3, stride=2, padding=1)
        self.hidden_layer = nn.Linear(self.conv_dims, hidden_dims)
        self.out_layer = nn.Linear(hidden_dims, embed_dims)

        self.scale = nn.Parameter(torch.zeros((1, hidden_dims)))
        self.bias = nn.Parameter(torch.zeros((1, hidden_dims)))

    def img_to_embed(self, z: torch.Tensor) -> torch.Tensor:
        z = self.in_conv(z)
        e = z.flatten(1)
        e = self.hidden_layer(e)
        e = e * (1 + self.scale) + self.bias
        e = self.out_layer(e)
        return e
    @torch.no_grad()
    def embed_to_img(self, e: torch.Tensor) -> torch.Tensor:
        e = inv_linear(e, self.out_layer)
        e = (e - self.bias) / (1 + self.scale)
        e = inv_linear(e, self.hidden_layer)
        z = e.view(-1, *self.conv_size)
        z = inv_conv2d(z, self.in_conv, self.img_size)
        return z

class ResBlock(nn.Module):
    def __init__(self, dims: int, kernel_size: int=5, stride: bool = False, hidden_dim_factor: float=2):
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