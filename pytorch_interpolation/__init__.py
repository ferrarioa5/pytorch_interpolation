import torch
from . import _C
from .ops import bilinear_interp, trilinear_interp_3d
from .interp import RegularGridInterpolator, RegularGridInterpolatorPyTorch, RegularGridInterpolator3D
from .grid_sample_interp import RegularGridInterpolatorGridSample
from .grid_sample_interp import RegularGridInterpolatorGridSample3D


class RegularGridInterpolatorAutomatic:
    """Unified interpolator: C++/CUDA for 2-D, grid_sample for 3-D.

    Automatically selects the backend based on ``len(points)``:

    * **2 axes** (default) → :class:`RegularGridInterpolator`
      (custom C++/CUDA kernel, supports ``method=0`` bilinear and
      ``method=1`` biquadratic).
    * **3 axes** → :class:`RegularGridInterpolatorGridSample3D`
      (``torch.nn.functional.grid_sample`` trilinear path — no custom
      build required).

    Args:
        points: tuple of 2 or 3 one-dimensional tensors defining the
                uniform grid axes.
        F:      field tensor — shape ``(nx, ny)`` for 2-D or
                ``(nx, ny, nz)`` for 3-D.
        fill_value:
            ``"nearest"`` → clamp / border padding
            ``None`` or ``0`` → zero for out-of-bounds queries
        method: interpolation method (only used for 2-D):
                ``0`` = bilinear, ``1`` = biquadratic.

    Usage — 2-D (default)::

        interp = RegularGridInterpolatorAutomatic(
            (x, y), F_2d, fill_value="nearest",
        )
        result = interp(xpt, ypt)

    Usage — 3-D::

        interp = RegularGridInterpolatorAutomatic(
            (x, y, z), F_3d, fill_value="nearest",
        )
        result = interp(xpt, ypt, zpt)
    """

    def __init__(self, points, F, fill_value=0.0, method=0):
        self._ndim = len(points)

        if self._ndim == 2:
            self._impl = RegularGridInterpolator(
                points, F, fill_value=fill_value, method=method,
            )
        elif self._ndim == 3:
            self._impl = RegularGridInterpolatorGridSample3D(
                points, F, fill_value=fill_value,
            )
        else:
            raise ValueError(
                f"Expected 2 or 3 grid axes, got {self._ndim}"
            )

    # -- expose common attributes / properties -------------------------

    @property
    def ndim(self):
        """Number of spatial dimensions (2 or 3)."""
        return self._ndim

    @property
    def F(self):
        return self._impl.F

    @F.setter
    def F(self, value):
        self._impl.F = value

    @property
    def x(self):
        return self._impl.x

    @property
    def y(self):
        return self._impl.y

    @property
    def z(self):
        if self._ndim < 3:
            raise AttributeError("z axis is not available in 2-D mode")
        return self._impl.z

    # -- interpolation --------------------------------------------------

    def __call__(self, *args):
        """Interpolate at query points.

        Pass ``(xpt, ypt)`` for 2-D or ``(xpt, ypt, zpt)`` for 3-D.
        """
        if len(args) != self._ndim:
            raise ValueError(
                f"Expected {self._ndim} coordinate arrays, got {len(args)}"
            )
        return self._impl(*args)