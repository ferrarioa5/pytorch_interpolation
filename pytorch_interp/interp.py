
import torch
from pytorch_interp import bilinear_interp


class RegularGridInterpolator:

    def __init__(self, points, F):

        assert isinstance(points, tuple)
        (x,y) = points
        self.x = x
        self.y = y
        
        assert isinstance(F, torch.Tensor)
        self.F = F
        self.G = torch.zeros(2**10)

        self.dx = float(x[1]-x[0])
        self.dy = float(y[1]-y[0])
        self.M1 = len(x)
        self.M2 = len(y)

    def __call__(self, xpt, ypt):
        return bilinear_interp(
            self.F, 
            self.x, self.y,
            xpt, ypt,
            self.M1, self.M2,
            self.dx, self.dy,
        )
