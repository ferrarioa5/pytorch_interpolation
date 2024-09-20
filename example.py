
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
x1  = -4.23
x2  = 12.6
y1  = -2.3
y2  = 2.2
x   = torch.linspace(x1,x2,M1).to(device)
y   = torch.linspace(y1,y2,M2).to(device)
xpt = (x2-x1)*torch.rand(N)+x1
xpt = xpt.to(device)
ypt = (y2-y1)*torch.rand(N)+y1
ypt = ypt.to(device)
X, Y = torch.meshgrid(x, y, indexing="ij")

F=torch.sin(X)*torch.sin(Y)

# scipy implementation
interp1 = scipy_rgi((x.cpu().numpy(), y.cpu().numpy()), F.cpu().numpy(),bounds_error=False, fill_value=0) 
G1 = interp1(np.array([xpt.cpu().numpy(), ypt.cpu().numpy()]).T)

# implementation
interp2 = my_rgi((x, y), F, fill_value=0)
G2 = interp2(xpt,ypt)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))

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
ax[2].set_title("pytorch_interpolation interpolation")
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
ax[2].set_xlim([xpt.cpu().numpy().min(),xpt.cpu().numpy().max()])
ax[2].set_ylim([ypt.cpu().numpy().min(),ypt.cpu().numpy().max()])

plt.savefig("example")


plt.show()


