
from pytorch_interpolation import RegularGridInterpolator as rgi
import torch
import numpy as np
import os
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

# bilinear implementation
interp1 = rgi((x, y), F, fill_value=fill_value, method=0)
G1 = interp1(xpt,ypt)

# biquadratic implementation
interp2 = rgi((x, y), F, fill_value=fill_value, method=1)
G2 = interp2(xpt,ypt)


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,6))
cmap = plt.cm.viridis

vmin,vmax = (fun(np.concatenate([F.cpu().numpy().flatten(),G1.cpu().numpy().flatten()])) for fun in (np.min,np.max))

ax[0, 0].contourf(X.cpu().numpy(), Y.cpu().numpy(), F.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
ax[0, 0].set_title("Real function")
ax[0, 0].set_xlabel("x")
ax[0, 0].set_ylabel("y")
ax[0, 0].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[0, 0].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])

ax[0, 1].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=G1.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
ax[0, 1].set_title("Bilinear interpolation")
ax[0, 1].set_xlabel("x")
ax[0, 1].set_ylabel("y")
ax[0, 1].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[0, 1].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])

ax[0, 2].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=G2.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
ax[0, 2].set_title("Biquadratic interpolation")
ax[0, 2].set_xlabel("x")
ax[0, 2].set_ylabel("y")
ax[0, 2].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[0, 2].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])



err1 = G1-function(xpt,ypt)
err2 = G2-function(xpt,ypt)

# leave the bottom-left subplot empty
ax[1,0].axis("off")

c1 = err1.cpu().numpy()
vmax1 = np.max(np.abs(c1))
sc1 = ax[1, 1].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=c1, cmap=plt.cm.seismic, vmin=-vmax1, vmax=vmax1)
ax[1, 1].set_title("Bilinear error (G1 - true)")
ax[1, 1].set_xlabel("x")
ax[1, 1].set_ylabel("y")
ax[1, 1].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[1, 1].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])
plt.colorbar(sc1, ax=ax[1, 1])

c2 = err2.cpu().numpy()
vmax2 = np.max(np.abs(c2))
sc2 = ax[1, 2].scatter(xpt.cpu().numpy(), ypt.cpu().numpy(), c=c2, cmap=plt.cm.seismic, vmin=-vmax2, vmax=vmax2)
ax[1, 2].set_title("Biquadratic error (G2 - true)")
ax[1, 2].set_xlabel("x")
ax[1, 2].set_ylabel("y")
ax[1, 2].set_xlim([xpt.cpu().numpy().min(), xpt.cpu().numpy().max()])
ax[1, 2].set_ylim([ypt.cpu().numpy().min(), ypt.cpu().numpy().max()])
plt.colorbar(sc2, ax=ax[1, 2])

plt.tight_layout()


print(f"Bilinear: max={err1.abs().max().item():.3e}, L2={err1.norm().item()/N:.3e} | "
    f"Biquadratic: max={err2.abs().max().item():.3e}, L2={err2.norm().item()/N:.3e}")


figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../figures/")
plt.savefig(os.path.join(figures_dir, "error_test"))
