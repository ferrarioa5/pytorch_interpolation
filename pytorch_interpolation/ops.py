import torch
from torch import Tensor

__all__ = ["bilinear_interp", "trilinear_interp_3d"]

def bilinear_interp(
                    F: Tensor, G: Tensor,
                    x: Tensor, y: Tensor,
                    xpt: Tensor, ypt: Tensor,
                    M1: int, M2: int,
                    dx: float, dy: float,
                    fill_method: int, fill_value:float,
                    method: int
                    ) -> Tensor:
    """Bilinear interpolation"""
    return torch.ops.extension_interp.bilinear_interp.default(F,G,x,y,xpt,ypt,M1,M2,dx,dy,fill_method,fill_value, method)

def trilinear_interp_3d(
                    F: Tensor, G: Tensor,
                    x: Tensor, y: Tensor, z: Tensor,
                    xpt: Tensor, ypt: Tensor, zpt: Tensor,
                    M1: int, M2: int, M3: int,
                    dx: float, dy: float, dz: float,
                    fill_method: int, fill_value: float,
                    ) -> Tensor:
    """Trilinear 3-D interpolation"""
    return torch.ops.extension_interp.trilinear_interp_3d.default(F,G,x,y,z,xpt,ypt,zpt,M1,M2,M3,dx,dy,dz,fill_method,fill_value)
