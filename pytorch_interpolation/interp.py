
from pytorch_interpolation import bilinear_interp
from pytorch_interpolation.ops import trilinear_interp_3d
import torch

class RegularGridInterpolator:

    def __init__(self, points, F, fill_value=0.0, method=0):

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
        self.method = method
        if type(fill_value)==float:
            self.fill_method=1
        elif type(fill_value)==int:
            self.fill_method=1
            self.fill_value = float(fill_value)
        elif fill_value==None:
            self.fill_method=2
            self.fill_value=0.0 # need to be a float in the c++ call (not actually used)
        elif fill_value=="nearest":
            self.fill_method=3
            self.fill_value=0.0 # need to be a float in the c++ call (not actually used)
        else:
            raise Exception("Provide a floating, None or nearest fill_value")



    def __call__(self, xpt, ypt):
        G = torch.empty_like(xpt)
        bilinear_interp(
            self.F, G,
            self.x, self.y,
            xpt, ypt,
            self.M1, self.M2,
            self.dx, self.dy,
            self.fill_method,
            self.fill_value,
            self.method
        )
        return G



class RegularGridInterpolatorPyTorch:

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
        if type(fill_value)==float:
            self.fill_method=1
        elif type(fill_value)==int:
            self.fill_method=1
            self.fill_value = float(fill_value)
        elif fill_value==None:
            self.fill_method=2
            self.fill_value=0.0 # need to be a float in the c++ call (not actually used)
        elif fill_value=="nearest":
            self.fill_method=3
            self.fill_value=0.0 # need to be a float in the c++ call (not actually used)
        else:
            raise Exception("Provide a floating, None or nearest fill_value")



    def __call__(self, xpt, ypt):

        if self.fill_method == 1:

            ind_x = ((xpt - self.x[0]) / self.dx).floor().long()
            ind_xp = ind_x + 1
            ind_y = ((ypt - self.y[0]) / self.dy).floor().long()
            ind_yp = ind_y + 1

            # w11 = ((self.x[ind_xp] - xpt) * (self.y[ind_yp] - ypt))
            # w12 = ((self.x[ind_xp] - xpt) * (ypt - self.y[ind_y]))
            # w21 = ((xpt - self.x[ind_x]) * (self.y[ind_yp] - ypt))
            # w22 = ((xpt - self.x[ind_x]) * (ypt - self.y[ind_y]))

            # mask of points with all indices inside valid ranges
            mask_valid = (ind_x >= 0) & (ind_xp < self.M1) & (ind_y >= 0) & (ind_yp < self.M2)

            # numeric fill_value
            G = torch.where(
                mask_valid,
                (
                    ((self.x[ind_xp] - xpt) * (self.y[ind_yp] - ypt))* self.F[ind_x, ind_y] + \
                    ((self.x[ind_xp] - xpt) * (ypt - self.y[ind_y])) * self.F[ind_x, ind_yp] + \
                    ((xpt - self.x[ind_x]) * (self.y[ind_yp] - ypt)) * self.F[ind_xp, ind_y] + \
                    ((xpt - self.x[ind_x]) * (ypt - self.y[ind_y])) * self.F[ind_xp, ind_yp]
                ) / (self.dx*self.dy),
                self.fill_value
                )

        elif self.fill_method == 2:
            pass

        elif self.fill_method == 3:
            ind_x = ((xpt - self.x[0]) / self.dx).floor().long()
            ind_y = ((ypt - self.y[0]) / self.dy).floor().long()
            ind_xp = ind_x + 1
            ind_yp = ind_y + 1

            w11 = ((self.x[ind_xp] - xpt) * (self.y[ind_yp] - ypt))
            w12 = ((self.x[ind_xp] - xpt) * (ypt - self.y[ind_y]))
            w21 = ((xpt - self.x[ind_x]) * (self.y[ind_yp] - ypt))
            w22 = ((xpt - self.x[ind_x]) * (ypt - self.y[ind_y]))

            # mask of points with all indices inside valid ranges
            mask_valid = (ind_x >= 0) & (ind_xp < self.M1) & (ind_y >= 0) & (ind_yp < self.M2)

            G = torch.where(
                mask_valid,
                (w11* self.F[ind_x, ind_y] + w12 * self.F[ind_x, ind_yp] + \
                w21 * self.F[ind_xp, ind_y] + w22 * self.F[ind_xp, ind_yp])/(self.dx*self.dy),
                self.F[ind_x.clamp(0, self.M1-1), ind_y.clamp(0, self.M2-1)]
                )

        return G


# ======================================================================
#  3-D  trilinear  — custom C++/CUDA kernel
# ======================================================================

class RegularGridInterpolator3D:
    """GPU-accelerated 3-D trilinear interpolation via custom CUDA kernel.

    Same API as RegularGridInterpolatorGridSample3D but calls the fused
    C++/CUDA ``trilinear_interp_3d`` kernel directly — no coordinate
    normalisation, no 5-D reshape, no grid_sample overhead.

    Args:
        points: tuple of (x, y, z) — three 1-D tensors for the uniform
                grid axes (lengths M1, M2, M3).
        F:      3-D tensor of shape ``(M1, M2, M3)`` with grid values.
                Stored row-major ``F[ix, iy, iz]`` in the kernel.
        fill_value:
            ``float`` or ``int`` → constant padding  (fill_method=1)
            ``None``             → linear extrapolation (fill_method=2)
            ``"nearest"``        → border / nearest clamp (fill_method=3)
    """

    def __init__(self, points, F, fill_value=0.0):
        assert isinstance(points, tuple) and len(points) == 3
        x, y, z = points
        self.x  = x.contiguous()
        self.y  = y.contiguous()
        self.z  = z.contiguous()
        self.dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        self.dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        self.dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
        self.M1 = len(x)
        self.M2 = len(y)
        self.M3 = len(z)
        self.F  = F.contiguous()

        # Resolve fill_method / fill_value the same way as 2-D
        if isinstance(fill_value, (float, int)):
            self.fill_method = 1
            self.fill_value  = float(fill_value)
        elif fill_value is None:
            self.fill_method = 2
            self.fill_value  = 0.0
        elif fill_value == "nearest":
            self.fill_method = 3
            self.fill_value  = 0.0
        else:
            raise ValueError("fill_value must be a float, None, or 'nearest'")

    def __call__(self, xpt, ypt, zpt):
        G = torch.empty_like(xpt)
        trilinear_interp_3d(
            self.F, G,
            self.x, self.y, self.z,
            xpt, ypt, zpt,
            self.M1, self.M2, self.M3,
            self.dx, self.dy, self.dz,
            self.fill_method,
            self.fill_value,
        )
        return G


