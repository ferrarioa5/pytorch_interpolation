


from pytorch_interp import bilinear_interp
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time
import matplotlib.pyplot as plt

N1 = 5000
N2 = 5000

def time_cpu_pytorch_interp(N):
    device = "cpu"
    torch.set_num_threads(8)
    x      = torch.linspace(0,1,N1).to(device)
    y      = torch.linspace(0,1,N2).to(device)
    xpt    = torch.rand(N).to(device)
    ypt    = torch.rand(N).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    start  = time.time()
    G      = bilinear_interp(F,x,y,xpt,ypt)
    return time.time()-start

def time_cuda_pytorch_interp(N):
    device = "cuda"
    start  = torch.cuda.Event(enable_timing=True)
    end    = torch.cuda.Event(enable_timing=True)
    x      = torch.linspace(0,1,N1).to(device)
    y      = torch.linspace(0,1,N2).to(device)
    xpt    = torch.rand(N).to(device)
    ypt    = torch.rand(N).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    start.record()
    G      = bilinear_interp(F,x,y,xpt,ypt)
    end.record()
    torch.cuda.synchronize()
    return 1e-3*(start.elapsed_time(end)) # conversion to sec

def time_scipy_interp(N):
    x      = np.linspace(0,1,N1)
    y      = np.linspace(0,1,N2)
    xpt    = np.random.rand(N)
    ypt    = np.random.rand(N)
    X, Y   = np.meshgrid(x, y, indexing="ij")
    F      = np.sin(X)*np.sin(Y)
    interp = RegularGridInterpolator((x, y), F, method='linear') #, bounds_error=False, fill_value=None)
    start  = time.time()
    G_rgi  = interp(np.array([xpt, ypt]).T)
    return time.time()-start


Ns   = 2**np.arange(15,24)
time_pytorch_cpu = []
time_pytorch_cuda = []
time_scipy = []

for N in Ns:
    time_pytorch_cpu.append(time_cpu_pytorch_interp(N))
    time_pytorch_cuda.append(time_cuda_pytorch_interp(N))
    time_scipy.append(time_scipy_interp(N))

ms=20

plt.figure(figsize=(5,5))
plt.plot(Ns, time_pytorch_cpu, 'g.', label="pytorch CPU", ms=ms)
plt.plot(Ns, time_pytorch_cuda, 'b.', label="pytorch GPU", ms=ms)
plt.plot(Ns, time_scipy, 'r.', label="scipy", ms=ms)
plt.xlabel("Number of interpolated points")
plt.ylabel("Time (sec)")
plt.legend()
plt.savefig("performance")
plt.show()
