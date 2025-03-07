
from pytorch_interp import RegularGridInterpolator as my_rgi
from scipy.interpolate import RegularGridInterpolator as scipy_rgi
import torch
import numpy as np
import matplotlib.pyplot as plt

device="cuda"
torch.set_num_threads(8)

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
F=X**2+Y**2-1


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
vmin,vmax = (fun(F.cpu().numpy()) for fun in (np.min,np.max))
cm = plt.cm.get_cmap('viridis')
ax[0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=cm,vmin=vmin,vmax=vmax)
ax[0].set_title("Real function")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_xlim([x1,x2])
ax[0].set_xlim([y1,y2])


x1=x1-10
y1=y1-10
x2=x2+10
y2=y2+10
xpt = (x2-x1)*torch.rand(N)+x1
xpt = xpt.to(device)
ypt = (y2-y1)*torch.rand(N)+y1
ypt = ypt.to(device)


# implementation
interp2 = my_rgi((x, y), F, fill_value="nearest")
G2 = interp2(xpt,ypt)
vmin,vmax = (fun(np.concatenate([F.cpu().numpy().flatten(),G2.cpu().numpy().flatten()])) for fun in (np.min,np.max))

ax[1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G2.cpu().numpy(),cmap=cm,vmin=vmin,vmax=vmax)
ax[1].set_title("pytorch_interpolation interpolation")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_xlim([xpt.cpu().numpy().min(),xpt.cpu().numpy().max()])
ax[1].set_ylim([ypt.cpu().numpy().min(),ypt.cpu().numpy().max()])


plt.savefig("nearest_extrapolation")


plt.show()


