


from pytorch_interp import RegularGridInterpolator
import torch
import numpy as np
import scipy.interpolate as scipy_interpolate
import torch_interpolations as torch_interpolate
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
    interp = RegularGridInterpolator((x,y),F)
    start  = time.time()
    G      = interp(xpt, ypt)
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
    torch.cuda.synchronize()
    interp      = RegularGridInterpolator((x,y),F)
    start.record()
    G  = interp(xpt, ypt)
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
    interp = scipy_interpolate.RegularGridInterpolator((x, y), F)
    start  = time.time()
    G_rgi  = interp(np.array([xpt, ypt]).T)
    return time.time()-start

def time_cpu_torch_interpolations(N):
    device = "cpu"
    torch.set_num_threads(8)
    x      = torch.linspace(0,1,N1).to(device)
    y      = torch.linspace(0,1,N2).to(device)
    xpt    = torch.rand(N).to(device)
    ypt    = torch.rand(N).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    interp = torch_interpolate.RegularGridInterpolator((x, y), F)
    start  = time.time()
    G_rgi  = interp([xpt, ypt])
    return time.time()-start

def time_cuda_torch_interpolations(N):
    device = "cuda"
    start  = torch.cuda.Event(enable_timing=True)
    end    = torch.cuda.Event(enable_timing=True)
    x      = torch.linspace(0,1,N1).to(device)
    y      = torch.linspace(0,1,N2).to(device)
    xpt    = torch.rand(N).to(device)
    ypt    = torch.rand(N).to(device)
    X, Y   = torch.meshgrid(x, y, indexing="ij")
    F      = torch.sin(X)*torch.sin(Y)
    interp = torch_interpolate.RegularGridInterpolator((x, y), F)
    torch.cuda.synchronize()
    start.record()
    G_rgi  = interp([xpt, ypt])
    end.record()
    torch.cuda.synchronize()
    return 1e-3*(start.elapsed_time(end)) # conversion to sec


Ns   = 2**torch.arange(10,26)
time_pytorch_cpu = []
time_pytorch_cuda = []
time_scipy = []
time_torch_interpolations_cpu = []
time_torch_interpolations_gpu = []

with torch.autograd.profiler.profile(use_device="cuda") as prof:
    for N in Ns:
        time_pytorch_cpu.append(time_cpu_pytorch_interp(N))
        time_pytorch_cuda.append(time_cuda_pytorch_interp(N))
        time_scipy.append(time_scipy_interp(N))
        time_torch_interpolations_cpu.append(time_cpu_torch_interpolations(N))
        time_torch_interpolations_gpu.append(time_cuda_torch_interpolations(N))
print(prof)




# ==== plotting ====
plt.figure(figsize=(5,10))
plt.subplot(2,1,1)
plot_type = plt.loglog
ms=20
plot_type(Ns, time_pytorch_cpu, 'g.', label="pytorch_interpolation CPU (OUR)", ms=ms)
plot_type(Ns, time_pytorch_cuda, 'b.', label="pytorch_interpolation GPU (OUR)", ms=ms)
plot_type(Ns, time_scipy, 'r.', label="scipy", ms=ms)
plot_type(Ns, time_torch_interpolations_cpu, 'y.', label="torch_interpolations CPU", ms=ms)
plot_type(Ns, time_torch_interpolations_gpu, 'k.', label="torch_interpolations GPU", ms=ms)
plt.xlabel("Number of interpolated points")
plt.ylabel("Execution time (sec)")
plt.legend()
plt.title("Loglog")
plt.xscale("log", base=2)
plt.yscale("log", base=2)

plt.subplot(2,1,2)
plot_type = plt.semilogx
ms=20
plot_type(Ns, time_pytorch_cpu, 'g.', label="pytorch_interpolation CPU (OUR)", ms=ms)
plot_type(Ns, time_pytorch_cuda, 'b.', label="pytorch_interpolation GPU (OUR)", ms=ms)
plot_type(Ns, time_scipy, 'r.', label="scipy", ms=ms)
plot_type(Ns, time_torch_interpolations_cpu, 'y.', label="torch_interpolations CPU", ms=ms)
plot_type(Ns, time_torch_interpolations_gpu, 'k.', label="torch_interpolations GPU", ms=ms)
plt.xlabel("Number of interpolated points")
plt.ylabel("Execution time (sec)")
plt.legend()
plt.title("Semilogx")
plt.xscale("log", base=2)
plt.savefig("performance")

plt.show()

