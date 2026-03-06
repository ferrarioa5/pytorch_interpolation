
import os
import sys

# Fix sys.path so the installed C++/CUDA extension is found
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_script_dir)
_saved_path = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) not in (_script_dir, _repo_dir)]

from pytorch_interpolation import RegularGridInterpolator as my_rgi
from pytorch_interpolation import RegularGridInterpolatorGridSample as gs_rgi

sys.path = _saved_path

import torch
import numpy as np
import matplotlib.pyplot as plt

device="cuda"
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)

M1  = 2**5
M2  = 2**5
N   = 2**12
x1  = -4
x2  = 4
y1  = -4
y2  = 4
x   = torch.linspace(x1,x2,M1).to(device)
y   = torch.linspace(y1,y2,M2).to(device)
X, Y = torch.meshgrid(x, y, indexing="ij")
function = lambda x,y: x**2 + y**2 -1
F=function(X,Y)

x1=x1-3
y1=y1-3
x2=x2+3
y2=y2+3
xpt = (x2-x1)*torch.rand(N)+x1
xpt = xpt.to(device)
ypt = (y2-y1)*torch.rand(N)+y1
ypt = ypt.to(device)


# --- fill_value mapping for grid_sample ---
# original: 0 → padding, "nearest" → border, None → linear_extrap
# grid_sample: 0 → padding_mode='zeros', "nearest" → padding_mode='border', None → padding_mode='zeros' (no linear extrap)
gs_fill_map = {0: 0.0, "nearest": "nearest", None: 0.0}

fill_values=[0,"nearest",None]
fill_labels=["fill=0 (padding)", 'fill="nearest" (border)', "fill=None (linear extrap)"]

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(24,18))
vmin,vmax = (fun(F.cpu().numpy()) for fun in (np.min,np.max))
cm = plt.cm.get_cmap('viridis')

# Row 0: Original bilinear
ax[0,0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=cm,vmin=vmin,vmax=vmax)
ax[0,0].set_title("Real function")
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel("Orig bilinear")
ax[0,0].set_xlim([x1,x2])
ax[0,0].set_ylim([y1,y2])

for i, fill_value in enumerate(fill_values):
    interp = my_rgi((x, y), F, fill_value=fill_value, method=0)
    G = interp(xpt,ypt)
    vmin_,vmax_ = (fun(np.concatenate([F.cpu().numpy().flatten(),G.cpu().numpy().flatten()])) for fun in (np.min,np.max))
    ax[0,i+1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G.cpu().numpy(),cmap=cm,vmin=vmin_,vmax=vmax_, s=2)
    ax[0,i+1].set_title(fill_labels[i])
    ax[0,i+1].set_xlim([x1,x2])
    ax[0,i+1].set_ylim([y1,y2])

# Row 1: Original biquadratic
ax[1,0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=cm,vmin=vmin,vmax=vmax)
ax[1,0].set_title("Real function")
ax[1,0].set_xlabel("x")
ax[1,0].set_ylabel("Orig biquadratic")
ax[1,0].set_xlim([x1,x2])
ax[1,0].set_ylim([y1,y2])

for i, fill_value in enumerate(fill_values):
    interp = my_rgi((x, y), F, fill_value=fill_value, method=1)
    G = interp(xpt,ypt)
    vmin_,vmax_ = (fun(np.concatenate([F.cpu().numpy().flatten(),G.cpu().numpy().flatten()])) for fun in (np.min,np.max))
    ax[1,i+1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G.cpu().numpy(),cmap=cm,vmin=vmin_,vmax=vmax_, s=2)
    ax[1,i+1].set_title(fill_labels[i])
    ax[1,i+1].set_xlim([x1,x2])
    ax[1,i+1].set_ylim([y1,y2])

# Row 2: grid_sample bilinear
ax[2,0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=cm,vmin=vmin,vmax=vmax)
ax[2,0].set_title("Real function")
ax[2,0].set_xlabel("x")
ax[2,0].set_ylabel("gs bilinear")
ax[2,0].set_xlim([x1,x2])
ax[2,0].set_ylim([y1,y2])

for i, fill_value in enumerate(fill_values):
    gs_fv = gs_fill_map[fill_value]
    interp = gs_rgi((x, y), F, fill_value=gs_fv, method=0)
    G = interp(xpt,ypt)
    vmin_,vmax_ = (fun(np.concatenate([F.cpu().numpy().flatten(),G.cpu().numpy().flatten()])) for fun in (np.min,np.max))
    ax[2,i+1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G.cpu().numpy(),cmap=cm,vmin=vmin_,vmax=vmax_, s=2)
    ax[2,i+1].set_title(fill_labels[i])
    ax[2,i+1].set_xlim([x1,x2])
    ax[2,i+1].set_ylim([y1,y2])

# Row 3: grid_sample bicubic
ax[3,0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=cm,vmin=vmin,vmax=vmax)
ax[3,0].set_title("Real function")
ax[3,0].set_xlabel("x")
ax[3,0].set_ylabel("gs bicubic")
ax[3,0].set_xlim([x1,x2])
ax[3,0].set_ylim([y1,y2])

for i, fill_value in enumerate(fill_values):
    gs_fv = gs_fill_map[fill_value]
    interp = gs_rgi((x, y), F, fill_value=gs_fv, method=1)
    G = interp(xpt,ypt)
    vmin_,vmax_ = (fun(np.concatenate([F.cpu().numpy().flatten(),G.cpu().numpy().flatten()])) for fun in (np.min,np.max))
    ax[3,i+1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G.cpu().numpy(),cmap=cm,vmin=vmin_,vmax=vmax_, s=2)
    ax[3,i+1].set_title(fill_labels[i])
    ax[3,i+1].set_xlim([x1,x2])
    ax[3,i+1].set_ylim([y1,y2])

plt.tight_layout()

figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../figures/")
plt.savefig(os.path.join(figures_dir, "extrapolation"))





