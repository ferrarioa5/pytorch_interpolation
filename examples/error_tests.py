
import os
import sys

# Fix sys.path so the installed C++/CUDA extension is found
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_script_dir)
_saved_path = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) not in (_script_dir, _repo_dir)]

from pytorch_interpolation import RegularGridInterpolator as rgi
from pytorch_interpolation import RegularGridInterpolatorGridSample as gs_rgi

sys.path = _saved_path

import torch
import numpy as np
import matplotlib.pyplot as plt

device="cuda"
dtype=torch.float32
torch.set_num_threads(8)

M1  = 2**9
M2  = 2**9
N   = 2**13
x1  = -4.23
x2  = 12.6
y1  = -2.3
y2  = 2.2
x   = torch.linspace(x1,x2,M1,dtype=dtype).to(device)
y   = torch.linspace(y1,y2,M2,dtype=dtype).to(device)

dx=float(x[1]-x[0])
dy=float(y[1]-y[0])
xmin=x1+dx
xmax=x2-dx
ymin=y1+dy
ymax=y2-dy

print(f"x in [{xmin},{xmax}] | y in [{ymin},{ymax}]")


xpt = (xmax-xmin)*torch.rand(N,dtype=dtype)+xmin
xpt = xpt.to(device)
ypt = (ymax-ymin)*torch.rand(N,dtype=dtype)+ymin
ypt = ypt.to(device)
X, Y = torch.meshgrid(x, y, indexing="ij")

function = lambda x,y: torch.sin(x)*torch.sin(y)
F=function(X,Y)

fill_value=0

# bilinear implementation (original)
interp1 = rgi((x, y), F, fill_value=fill_value, method=0)
G1 = interp1(xpt,ypt)

# biquadratic implementation (original)
interp2 = rgi((x, y), F, fill_value=fill_value, method=1)
G2 = interp2(xpt,ypt)

# grid_sample bilinear
interp3 = gs_rgi((x, y), F, fill_value=float(fill_value), method=0)
G3 = interp3(xpt, ypt)


fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(24,12))
cmap = plt.cm.viridis

vmin,vmax = (fun(np.concatenate([F.cpu().numpy().flatten(),G1.cpu().numpy().flatten()])) for fun in (np.min,np.max))

# Row 0: interpolated values
ax[0, 0].contourf(X.cpu().numpy(), Y.cpu().numpy(), F.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
ax[0, 0].set_title("Real function")
ax[0, 0].set_xlabel("x")
ax[0, 0].set_ylabel("y")
ax[0, 0].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[0, 0].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])

ax[0, 1].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=G1.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax, s=2)
ax[0, 1].set_title("Original bilinear")
ax[0, 1].set_xlabel("x")
ax[0, 1].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[0, 1].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])

ax[0, 2].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=G2.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax, s=2)
ax[0, 2].set_title("Original biquadratic")
ax[0, 2].set_xlabel("x")
ax[0, 2].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[0, 2].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])

ax[0, 3].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=G3.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax, s=2)
ax[0, 3].set_title("grid_sample bilinear")
ax[0, 3].set_xlabel("x")
ax[0, 3].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[0, 3].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])

# Row 1: errors
exact = function(xpt, ypt)
err1 = G1 - exact
err2 = G2 - exact
err3 = G3 - exact

ax[1, 0].axis("off")

c1 = err1.cpu().numpy()
vmax1 = np.max(np.abs(c1))
sc1 = ax[1, 1].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=c1, cmap=plt.cm.seismic, vmin=-vmax1, vmax=vmax1, s=2)
ax[1, 1].set_title(f"Orig bilinear err (max={err1.abs().max().item():.2e})")
ax[1, 1].set_xlabel("x")
ax[1, 1].set_ylabel("y")
ax[1, 1].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[1, 1].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])
plt.colorbar(sc1, ax=ax[1, 1])

c2 = err2.cpu().numpy()
vmax2 = np.max(np.abs(c2))
sc2 = ax[1, 2].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=c2, cmap=plt.cm.seismic, vmin=-vmax2, vmax=vmax2, s=2)
ax[1, 2].set_title(f"Orig biquad err (max={err2.abs().max().item():.2e})")
ax[1, 2].set_xlabel("x")
ax[1, 2].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[1, 2].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])
plt.colorbar(sc2, ax=ax[1, 2])

c3 = err3.cpu().numpy()
vmax3 = np.max(np.abs(c3))
sc3 = ax[1, 3].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=c3, cmap=plt.cm.seismic, vmin=-vmax3, vmax=vmax3, s=2)
ax[1, 3].set_title(f"gs bilinear err (max={err3.abs().max().item():.2e})")
ax[1, 3].set_xlabel("x")
ax[1, 3].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[1, 3].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])
plt.colorbar(sc3, ax=ax[1, 3])

# Row 2: difference between original and grid_sample
ax[2, 0].axis("off")
ax[2, 1].axis("off")

diff_bilinear = (G1 - G3).cpu().numpy()
vmax_d1 = np.max(np.abs(diff_bilinear)) if np.max(np.abs(diff_bilinear)) > 0 else 1e-10
sc_d1 = ax[2, 2].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=diff_bilinear, cmap=plt.cm.seismic, vmin=-vmax_d1, vmax=vmax_d1, s=2)
ax[2, 2].set_title(f"orig-gs bilinear (max={np.max(np.abs(diff_bilinear)):.2e})")
ax[2, 2].set_xlabel("x")
ax[2, 2].set_ylabel("y")
ax[2, 2].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[2, 2].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])
plt.colorbar(sc_d1, ax=ax[2, 2])

ax[2, 3].axis("off")

plt.tight_layout()


print(f"Orig  bilinear    : max={err1.abs().max().item():.3e}, L2={err1.norm().item()/N:.3e}")
print(f"Orig  biquadratic : max={err2.abs().max().item():.3e}, L2={err2.norm().item()/N:.3e}")
print(f"gs    bilinear    : max={err3.abs().max().item():.3e}, L2={err3.norm().item()/N:.3e}")


figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../figures/")
plt.savefig(os.path.join(figures_dir, "error_test"))
