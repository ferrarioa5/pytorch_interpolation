import torch
from torch import Tensor

__all__ = ["bilinear_interp"]

def bilinear_interp(
                    F: Tensor, G: Tensor, 
                    x: Tensor, y: Tensor,
                    xpt: Tensor, ypt: Tensor,
                    M1: int, M2: int,
                    dx: float, dy: float
                    ) -> Tensor:
    """Bilinear interpolation"""
    return torch.ops.extension_interp.bilinear_interp.default(F,G,x,y,xpt,ypt,M1,M2,dx,dy)
