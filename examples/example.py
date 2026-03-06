
import os
import sys

# Fix sys.path so the installed C++/CUDA extension is found, not the local source dir
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_script_dir)
_saved_path = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) not in (_script_dir, _repo_dir)]

from pytorch_interpolation import RegularGridInterpolator as my_rgi
from pytorch_interpolation import RegularGridInterpolatorGridSample as gs_rgi

sys.path = _saved_path

from scipy.interpolate import RegularGridInterpolator as scipy_rgi
import torch
import numpy as np
import matplotlib.pyplot as plt

device="cuda"
dtype=torch.float32
torch.set_num_threads(8)

M1  = 2**8
M2  = 2**8
N   = 2**13
x1  = -4.23
x2  = 12.6
y1  = -2.3
y2  = 2.2
x   = torch.linspace(x1,x2,M1,dtype=dtype).to(device)
y   = torch.linspace(y1,y2,M2,dtype=dtype).to(device)
xpt = (x2-x1)*torch.rand(N,dtype=dtype)+x1
xpt = xpt.to(device)
ypt = (y2-y1)*torch.rand(N,dtype=dtype)+y1
ypt = ypt.to(device)
X, Y = torch.meshgrid(x, y, indexing="ij")

function = lambda x,y: torch.sin(x)*torch.sin(y)
F=function(X,Y)

# scipy implementation
interp1 = scipy_rgi((x.cpu().numpy(), y.cpu().numpy()), F.cpu().numpy(),bounds_error=False, fill_value=0)
G1 = interp1(np.array([xpt.cpu().numpy(), ypt.cpu().numpy()]).T)

# original C++/CUDA implementation
interp2 = my_rgi((x, y), F, fill_value=0, method=1)
G2 = interp2(xpt,ypt)

# grid_sample implementation (bicubic)
interp3 = gs_rgi((x, y), F, fill_value=0.0, method=1)
G3 = interp3(xpt, ypt)


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(24,6))

vmin,vmax = (fun(np.concatenate([F.cpu().numpy().flatten(),G1])) for fun in (np.min,np.max))

ax[0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[0].set_title("Real function")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_xlim([xpt.cpu().numpy().min(),xpt.cpu().numpy().max()])
ax[0].set_ylim([ypt.cpu().numpy().min(),ypt.cpu().numpy().max()])


ax[1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G1,cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[1].set_title("Scipy interpolation")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_xlim([xpt.cpu().numpy().min(),xpt.cpu().numpy().max()])
ax[1].set_ylim([ypt.cpu().numpy().min(),ypt.cpu().numpy().max()])

ax[2].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G2.cpu().numpy(),cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[2].set_title("pytorch_interpolation (biquadratic)")
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
ax[2].set_xlim([xpt.cpu().numpy().min(),xpt.cpu().numpy().max()])
ax[2].set_ylim([ypt.cpu().numpy().min(),ypt.cpu().numpy().max()])

ax[3].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G3.cpu().numpy(),cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[3].set_title("grid_sample (bicubic)")
ax[3].set_xlabel("x")
ax[3].set_ylabel("y")
ax[3].set_xlim([xpt.cpu().numpy().min(),xpt.cpu().numpy().max()])
ax[3].set_ylim([ypt.cpu().numpy().min(),ypt.cpu().numpy().max()])



figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../figures/")
plt.savefig(os.path.join(figures_dir, "example"))






