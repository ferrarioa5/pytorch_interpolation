
import src as ext

import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

device="cpu"
# device="cuda"

M1  = 1000
M2  = 500
N   = 400
x1  = -1
x2  = 1
y1  = -1
y2  = 1
x   = torch.linspace(x1,x2,M1).to(device)
y   = torch.linspace(y1,y2,M2).to(device)
xpt = (x2-x1)*torch.rand(N)+x1
xpt=xpt.to(device)
ypt = (y2-y1)*torch.rand(N)+y1
ypt=ypt.to(device)
X, Y = torch.meshgrid(x, y)

F=torch.sin(X)*torch.sin(Y)

# F   = torch.zeros(M1,M2).to(device)
# F[torch.logical_and(X>0,Y>0)]=1


interp_spline = RegularGridInterpolator((x.cpu().numpy(), y.cpu().numpy()), F.cpu().numpy(), method='linear') #, bounds_error=False, fill_value=None)
G_rgi = interp_spline(np.array([xpt.cpu().numpy(), ypt.cpu().numpy()]).T)

G = ext.ops.bilinear_interp(F,x,y,xpt,ypt)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

vmin,vmax = (fun(np.concatenate([F.cpu().numpy().flatten(),G_rgi])) for fun in (np.min,np.max))

ax[0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[0].set_title("Real function")

ax[1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G.cpu().numpy(),cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[1].set_title("Torch interpolation")

ax[2].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G_rgi,cmap=plt.cm.viridis,vmin=vmin,vmax=vmax)
ax[2].set_title("Scipy interpolation")




plt.show()


