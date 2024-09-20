
from pytorch_interp import bilinear_interp
import torch

class RegularGridInterpolator:

    def __init__(self, points, F, fill_value=0.0):

        assert isinstance(points, tuple)
        (x,y) = points
        self.x = x
        self.y = y
        self.dx = float(x[1]-x[0])
        self.dy = float(y[1]-y[0])
        self.M1 = len(x)
        self.M2 = len(y)
        self.F = F
        self.fill_value = fill_value
        if fill_value==None:
            self.fill_method=2
            self.fill_value=0.0 # need to be a float in the c++ call (not actually used)
        elif type(fill_value)==float:
            self.fill_method=1
        elif type(fill_value)==int:
            self.fill_method=1
            self.fill_value = float(fill_value)
        else:
            raise("Provide a floating or None fill_value")
            


    def __call__(self, xpt, ypt):
        G = torch.zeros_like(xpt)
        # G = torch.empty(xpt.size(), dtype=torch.float32)
        bilinear_interp(
            self.F, G,
            self.x, self.y,
            xpt, ypt,
            self.M1, self.M2,
            self.dx, self.dy,
            self.fill_method,
            self.fill_value
        )
        return G
