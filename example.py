
from pytorch_interp import bilinear_interp

import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

device="cpu"

M1  = 1000
M2  = 500
N   = 400
x1  = -2
x2  = 12
y1  = -1.2
y2  = 6
x   = torch.linspace(x1,x2,M1).to(device)
y   = torch.linspace(y1,y2,M2).to(device)
xpt = (x2-x1)*torch.rand(N)+x1
xpt=xpt.to(device)
ypt = (y2-y1)*torch.rand(N)+y1
ypt=ypt.to(device)
X, Y = torch.meshgrid(x, y, indexing="ij")

F=torch.sin(X)*torch.sin(Y)

interp_spline = RegularGridInterpolator((x.cpu().numpy(), y.cpu().numpy()), F.cpu().numpy(), method='linear') #, bounds_error=False, fill_value=None)
G_rgi = interp_spline(np.array([xpt.cpu().numpy(), ypt.cpu().numpy()]).T)

G = bilinear_interp(F,x,y,xpt,ypt)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3))

vmin,vmax = (fun(np.concatenate([F.cpu().numpy().flatten(),G_rgi])) for fun in (np.min,np.max))

ax[0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[0].set_title("Real function")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

ax[1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G_rgi,cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[1].set_title("Scipy interpolation")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")

ax[2].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G.cpu().numpy(),cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[2].set_title("Torch interpolation")
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")

plt.savefig("example")


plt.show()


