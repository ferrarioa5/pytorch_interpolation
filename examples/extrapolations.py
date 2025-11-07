
import os
from pytorch_interpolation import RegularGridInterpolator as my_rgi
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


fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(18,6))
vmin,vmax = (fun(F.cpu().numpy()) for fun in (np.min,np.max))
cm = plt.cm.get_cmap('viridis')
ax[0,0].contourf(X.cpu().numpy(),Y.cpu().numpy(),F.cpu().numpy(),cmap=cm,vmin=vmin,vmax=vmax)
ax[0,0].set_title("Real function")
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel("y")
ax[0,0].set_xlim([x1,x2])
ax[0,0].set_ylim([y1,y2])


fill_values=[0,"nearest",None]

for i, fill_value in enumerate(fill_values):

    # implementation
    interp2 = my_rgi((x, y), F, fill_value=fill_value, method=0)
    G2 = interp2(xpt,ypt)
    vmin,vmax = (fun(np.concatenate([F.cpu().numpy().flatten(),G2.cpu().numpy().flatten()])) for fun in (np.min,np.max))

    ax[0,i+1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G2.cpu().numpy(),cmap=cm,vmin=vmin,vmax=vmax)
    ax[0,i+1].set_title(f"fill_value={fill_value}")
    ax[0,i+1].set_xlim([xpt.cpu().numpy().min(),xpt.cpu().numpy().max()])
    ax[0,i+1].set_ylim([ypt.cpu().numpy().min(),ypt.cpu().numpy().max()])
ax[0,1].set_ylabel("Bilinear")


for i, fill_value in enumerate(fill_values):

    # implementation
    interp2 = my_rgi((x, y), F, fill_value=fill_value, method=1)
    G2 = interp2(xpt,ypt)
    vmin,vmax = (fun(np.concatenate([F.cpu().numpy().flatten(),G2.cpu().numpy().flatten()])) for fun in (np.min,np.max))

    ax[1,i+1].scatter(xpt.cpu().numpy(),ypt.cpu().numpy(),c=G2.cpu().numpy(),cmap=cm,vmin=vmin,vmax=vmax)
    ax[1,i+1].set_title(f"fill_value={fill_value}")
    ax[1,i+1].set_xlabel("x")
    ax[1,i+1].set_xlim([xpt.cpu().numpy().min(),xpt.cpu().numpy().max()])
    ax[1,i+1].set_ylim([ypt.cpu().numpy().min(),ypt.cpu().numpy().max()])
ax[1,1].set_ylabel("Biquadratic")



figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../figures/")
plt.savefig(os.path.join(figures_dir, "extrapolation"))





