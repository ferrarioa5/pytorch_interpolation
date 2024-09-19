
from pytorch_interp import bilinear_interp
import torch

class RegularGridInterpolator:

    def __init__(self, points, F):

        assert isinstance(points, tuple)
        (x,y) = points
        self.x = x
        self.y = y
        self.dx = float(x[1]-x[0])
        self.dy = float(y[1]-y[0])
        self.M1 = len(x)
        self.M2 = len(y)
        self.F = F


    def __call__(self, xpt, ypt):
        G = torch.zeros_like(xpt)
        # G = torch.empty(xpt.size(), dtype=torch.float32)
        bilinear_interp(
            self.F, G,
            self.x, self.y,
            xpt, ypt,
            self.M1, self.M2,
            self.dx, self.dy,
        )
        return G
