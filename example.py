
from pytorch_interp import RegularGridInterpolator as my_rgi
from scipy.interpolate import RegularGridInterpolator as scipy_rgi

import torch
import numpy as np
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

device="cuda"

M1  = 1000
M2  = 500
N   = 2**15
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

# scipy implementation
interp_spline = scipy_rgi((x.cpu().numpy(), y.cpu().numpy()), F.cpu().numpy()) 
G1 = interp_spline(np.array([xpt.cpu().numpy(), ypt.cpu().numpy()]).T)

torch.cuda.empty_cache()

# our implementation
interp_spline = my_rgi((x, y), F)
G2 = interp_spline(xpt,ypt)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3))

vmin,vmax = (fun(np.concatenate([F.cpu().numpy().flatten(),G1])) for fun in (np.min,np.max))

ax[0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[0].set_title("Real function")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

ax[1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G1,cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[1].set_title("Scipy interpolation")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")

ax[2].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G2.cpu().numpy(),cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[2].set_title("pytorch_interpolation interpolation")
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")

plt.savefig("example")


plt.show()


