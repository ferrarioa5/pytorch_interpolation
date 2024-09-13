import torch
from torch import Tensor

__all__ = ["bilinear_interp"]

def bilinear_interp(F: Tensor, x: Tensor, y: Tensor,
                    xpt: Tensor, ypt: Tensor) -> Tensor:
    """Bilinear interpolation"""
    return torch.ops.extension_interp.bilinear_interp.default(F,x,y,xpt,ypt)
