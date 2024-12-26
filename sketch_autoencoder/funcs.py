import torch
import torch.nn as nn
import torch.nn.functional as F


    
@torch.jit.script
def sparsity_loss(feats: torch.Tensor, dec_weight: torch.Tensor) -> torch.Tensor:
    """

    Args:
        feats (torch.Tensor): NxF tensor
        dec_weight (torch.Tensor): LxF tensor

    Returns:
        torch.Tensor: _description_
    """
    norms = torch.linalg.vector_norm(dec_weight.T, dim=-1, ord=2, keepdim=True) # Fx1 tensor
    loss = feats @ norms # Nx1 tensor
    return loss.mean()