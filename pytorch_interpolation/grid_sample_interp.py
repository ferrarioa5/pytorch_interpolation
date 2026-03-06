"""RegularGridInterpolator backed by torch.nn.functional.grid_sample.

No custom C++/CUDA extension needed — works out of the box with any
PyTorch installation.  Supports 2-D (bilinear) and 3-D (trilinear)
interpolation on uniform grids.

Usage — 2-D:
    from pytorch_interpolation import RegularGridInterpolatorGridSample

    interp = RegularGridInterpolatorGridSample(
        (x_1d, y_1d), F_2d, fill_value="nearest",
    )
    result = interp(xpt, ypt)

Usage — 3-D:
    interp3d = RegularGridInterpolatorGridSample3D(
        (x_1d, y_1d, z_1d), F_3d, fill_value="nearest",
    )
    result = interp3d(xpt, ypt, zpt)
"""

import torch
import torch.nn.functional as TF


# ======================================================================
#  2-D  bilinear  (unchanged API)
# ======================================================================

class RegularGridInterpolatorGridSample:
    """GPU-accelerated 2-D bilinear interpolation via grid_sample.

    Same API as RegularGridInterpolator (method=0).

        fill_value="nearest"  →  border / clamp behaviour
        fill_value=None / 0   →  zero for out-of-bounds queries

    The .F setter performs a transpose + contiguous copy (~6 µs on
    GPU for 512×512) vs the original's plain attribute set (~0.1 µs).
    """

    def __init__(self, points, F, fill_value=0.0, method=0):
        x, y = points

        self._x0 = float(x[0])
        self._y0 = float(y[0])
        self._x_range = float(x[-1] - x[0])
        self._y_range = float(y[-1] - y[0])

        self.x = x
        self.y = y
        self.dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        self.dy = float(y[1] - y[0]) if len(y) > 1 else 1.0

        self._padding = self._resolve_padding(fill_value)
        self._F = F
        self._build_volume(F)

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _resolve_padding(fill_value):
        if isinstance(fill_value, str) and fill_value == "nearest":
            return 'border'
        return 'zeros'

    def _build_volume(self, F):
        # F is (nx, ny) → grid_sample wants (1, 1, ny, nx)
        self._volume = F.T.contiguous().unsqueeze(0).unsqueeze(0)

    # -- .F property ----------------------------------------------------

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value
        self._build_volume(value)

    # -- interpolation --------------------------------------------------

    def __call__(self, xpt, ypt):
        orig_shape = xpt.shape
        xf = xpt.reshape(-1)
        yf = ypt.reshape(-1)
        n = xf.shape[0]

        x_norm = 2.0 * (xf - self._x0) / self._x_range - 1.0
        y_norm = 2.0 * (yf - self._y0) / self._y_range - 1.0

        grid = torch.stack([x_norm, y_norm], dim=-1).reshape(1, 1, n, 2)

        result = TF.grid_sample(
            self._volume, grid,
            mode='bilinear',
            padding_mode=self._padding,
            align_corners=True,
        )
        return result.reshape(orig_shape)


# ======================================================================
#  3-D  trilinear
# ======================================================================

class RegularGridInterpolatorGridSample3D:
    """GPU-accelerated 3-D trilinear interpolation via grid_sample.

    Uses the 5-D path of ``torch.nn.functional.grid_sample``
    (input: ``(N, C, D, H, W)``, grid last dim = 3) which maps directly
    to cuDNN's volumetric sampling — no custom C++/CUDA code required.

    Args:
        points: tuple of (x, y, z) — three 1-D tensors defining the
                uniform grid axes (lengths nx, ny, nz).
        F:      3-D tensor of shape ``(nx, ny, nz)`` with the grid values.
        fill_value:
            ``"nearest"`` → clamp queries to boundary values
            ``None`` or ``0`` → zero for out-of-bounds queries

    Usage::

        interp = RegularGridInterpolatorGridSample3D(
            (x, y, z), F, fill_value="nearest",
        )
        result = interp(xpt, ypt, zpt)   # any shape → same shape out
    """

    def __init__(self, points, F, fill_value=0.0):
        x, y, z = points

        self._x0 = float(x[0])
        self._y0 = float(y[0])
        self._z0 = float(z[0])
        self._x_range = float(x[-1] - x[0])
        self._y_range = float(y[-1] - y[0])
        self._z_range = float(z[-1] - z[0])

        self.x = x
        self.y = y
        self.z = z
        self.dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        self.dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        self.dz = float(z[1] - z[0]) if len(z) > 1 else 1.0

        self._padding = self._resolve_padding(fill_value)
        self._F = F
        self._build_volume(F)

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _resolve_padding(fill_value):
        if isinstance(fill_value, str) and fill_value == "nearest":
            return 'border'
        return 'zeros'

    def _build_volume(self, F):
        """Convert (nx, ny, nz) → grid_sample's (1, 1, nz, ny, nx).

        grid_sample 5-D convention:
            input  – (N, C, D, H, W)
            grid   – (N, D_out, H_out, W_out, 3)
                     grid[..., 0] → W  (x)
                     grid[..., 1] → H  (y)
                     grid[..., 2] → D  (z)

        F is stored as F[ix, iy, iz]  →  permute to (iz, iy, ix).
        """
        self._volume = F.permute(2, 1, 0).contiguous().unsqueeze(0).unsqueeze(0)

    # -- .F property ----------------------------------------------------

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value
        self._build_volume(value)

    # -- interpolation --------------------------------------------------

    def __call__(self, xpt, ypt, zpt):
        """Trilinear interpolation at arbitrary query points.

        Args:
            xpt, ypt, zpt: tensors of **identical** shape.

        Returns:
            Tensor of the same shape with interpolated values.
        """
        orig_shape = xpt.shape
        xf = xpt.reshape(-1)
        yf = ypt.reshape(-1)
        zf = zpt.reshape(-1)
        n = xf.shape[0]

        # Normalise physical coords → [-1, 1]  (align_corners=True)
        x_norm = 2.0 * (xf - self._x0) / self._x_range - 1.0
        y_norm = 2.0 * (yf - self._y0) / self._y_range - 1.0
        z_norm = 2.0 * (zf - self._z0) / self._z_range - 1.0

        # grid: (1, 1, 1, n, 3)  — D_out=1, H_out=1, W_out=n
        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).reshape(1, 1, 1, n, 3)

        result = TF.grid_sample(
            self._volume, grid,
            mode='bilinear',           # 'bilinear' = trilinear for 5-D
            padding_mode=self._padding,
            align_corners=True,
        )
        # result shape: (1, 1, 1, 1, n) → original shape
        return result.reshape(orig_shape)
