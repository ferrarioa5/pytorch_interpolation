#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace extension_interp {

template <typename scalar_t>
__device__ void compute_G_k_BilinearCUDA( const int k,
        const scalar_t * xpts, const scalar_t * ypts,
        const scalar_t * x, const scalar_t * y,
        const int ind_x, const int ind_xp,
        const int ind_y, const int ind_yp,
        const scalar_t * F, const int M2,
        const double dx, const double dy,
        scalar_t& out
      ) {

  const scalar_t w11 = (x[ind_xp]-xpts[k])*(y[ind_yp]-ypts[k]);
  const scalar_t w12 = (x[ind_xp]-xpts[k])*(ypts[k]-y[ind_y]);
  const scalar_t w21 = (xpts[k]-x[ind_x])*(y[ind_yp]-ypts[k]);
  const scalar_t w22 = (xpts[k]-x[ind_x])*(ypts[k]-y[ind_y]);

  out = (w11*F[ind_x*M2+ind_y] + w12*F[ind_x*M2+ind_yp] + w21*F[ind_xp*M2+ind_y] + w22*F[ind_xp*M2+ind_yp])/(dx*dy);

}

template <typename scalar_t>
__device__ void compute_G_k_BIquadraticCUDA( const int k,
        const scalar_t * xpts, const scalar_t * ypts,
        const scalar_t * x, const scalar_t * y,
        const int ind_x, const int ind_xm, const int ind_xp,
        const int ind_y, const int ind_ym, const int ind_yp,
        const scalar_t * F, const int M2,
        const double dx, const double dy,
        scalar_t& out
      ) {

  if(ind_xm<0 || ind_ym<0){
    compute_G_k_BilinearCUDA(
        k, xpts, ypts, x, y,
        ind_x, ind_xp,
        ind_y, ind_yp,
        F, M2,
        dx, dy,
        out
      );
  }
  else {

    // using lagrange polynomials

    const scalar_t L0_x = (xpts[k]-x[ind_x])*(xpts[k]-x[ind_xp])/( 2*dx*dx );
    const scalar_t L1_x = (xpts[k]-x[ind_xm])*(xpts[k]-x[ind_xp])/( -dx*dx );
    const scalar_t L2_x = (xpts[k]-x[ind_xm])*(xpts[k]-x[ind_x])/( 2*dx*dx );

    const scalar_t L0_y = (ypts[k]-y[ind_y])*(ypts[k]-y[ind_yp])/( 2*dy*dy );
    const scalar_t L1_y = (ypts[k]-y[ind_ym])*(ypts[k]-y[ind_yp])/( -dy*dy );
    const scalar_t L2_y = (ypts[k]-y[ind_ym])*(ypts[k]-y[ind_y])/( 2*dy*dy );

    out = (L0_x*L0_y*F[ind_xm*M2+ind_ym] + L0_x*L1_y*F[ind_xm*M2+ind_y] + L0_x*L2_y*F[ind_xm*M2+ind_yp]
          + L1_x*L0_y*F[ind_x*M2+ind_ym]   + L1_x*L1_y*F[ind_x*M2+ind_y]   + L1_x*L2_y*F[ind_x*M2+ind_yp]
          + L2_x*L0_y*F[ind_xp*M2+ind_ym] + L2_x*L1_y*F[ind_xp*M2+ind_y] + L2_x*L2_y*F[ind_xp*M2+ind_yp]);

  }
}

template <typename scalar_t>
__global__ void biquadratic_interpolation_kernel_CUDA_padding(
  scalar_t* G, const scalar_t* F,
  const scalar_t* xpts, const scalar_t* ypts,
  const int M1, const int M2, const int N,
  const double dx, const double dy,
  const scalar_t* x, const scalar_t* y,
  const double fill_value) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    const int ind_x  = floor((xpts[k]-x[0])/dx);
    const int ind_xm = ind_x-1;
    const int ind_xp = ind_x+1;
    const int ind_y  = floor((ypts[k]-y[0])/dy);
    const int ind_ym = ind_y-1;
    const int ind_yp = ind_y+1;

    if ( 0 <= ind_x && ind_xp  < M1 && 0 <= ind_y && ind_yp  < M2 ) {
      compute_G_k_BIquadraticCUDA(
        k, xpts, ypts, x, y,
        ind_x, ind_xm,  ind_xp,
        ind_y, ind_ym,  ind_yp,
        F, M2,
        dx, dy,
        G[k]
      );
    }
    else{
      G[k] = fill_value;
    }
  }
}

template <typename scalar_t>
__global__ void biquadratic_interpolation_kernel_CUDA_linear_extrap_linear(
  scalar_t* G, const scalar_t* F,
  const scalar_t* xpts, const scalar_t* ypts,
  const int M1, const int M2, const int N,
  const double dx, const double dy,
  const scalar_t* x, const scalar_t* y) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    int ind_x  = floor((xpts[k]-x[0])/dx);
    int ind_y  = floor((ypts[k]-y[0])/dy);
    int ind_xm = ind_x-1;
    int ind_xp = ind_x+1;
    int ind_ym = ind_y-1;
    int ind_yp = ind_y+1;

    if (ind_xm<0) {
      ind_xm=0;
      ind_x=1;
      ind_xp=2;
    }
    if (ind_xp>=M1) {
      ind_xp=M1-1;
      ind_x=M1-2;
      ind_xm=M1-3;
    }
    if (ind_ym<0) {
      ind_ym=0;
      ind_y=1;
      ind_yp=2;
    }
    if (ind_yp>=M2) {
      ind_yp=M2-1;
      ind_y=M2-2;
      ind_ym=M2-3;
    }

    compute_G_k_BIquadraticCUDA(
      k, xpts, ypts, x, y,
      ind_x, ind_xm,  ind_xp,
      ind_y, ind_ym,  ind_yp,
      F, M2,
      dx, dy,
      G[k]
    );

    }
  }

template <typename scalar_t>
__global__ void biquadratic_interpolation_kernel_CUDA_linear_extrap_nearest(
  scalar_t * G, const scalar_t * F,
  const scalar_t * xpts, const scalar_t * ypts,
  const int M1, const int M2, const int N,
  const double dx, const double dy,
  const scalar_t * x, const scalar_t * y) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    int ind_x  = floor((xpts[k]-x[0])/dx);
    int ind_y  = floor((ypts[k]-y[0])/dy);
    int ind_xm = ind_x-1;
    int ind_xp = ind_x+1;
    int ind_ym = ind_y-1;
    int ind_yp = ind_y+1;

    if ( 0 <= ind_xm && ind_xp  < M1 && 0 <= ind_ym && ind_yp  < M2 ) {

      compute_G_k_BIquadraticCUDA(
        k, xpts, ypts, x, y,
        ind_x, ind_xm,  ind_xp,
        ind_y, ind_ym,  ind_yp,
        F, M2,
        dx, dy,
        G[k]
      );

    }
    else {
      if (ind_xm<0) {
        ind_x=0;
      }
      if (ind_x>=M1) {
        ind_x=M1-1;
      }
      if (ind_ym<0) {
        ind_y=0;
      }
      if (ind_y>=M2) {
        ind_y=M2-1;
      }
      G[k]=F[ind_x*M2+ind_y];
    }
  }
}

template <typename scalar_t>
__global__ void bilinear_interpolation_kernel_CUDA_padding(
  scalar_t * G, const scalar_t * F,
  const scalar_t * xpts, const scalar_t * ypts,
  const int M1, const int M2, const int N,
  const double dx, const double dy,
  const scalar_t * x, const scalar_t * y,
  const double fill_value) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    const int ind_x  = floor((xpts[k]-x[0])/dx);
    const int ind_xp = ind_x+1;
    const int ind_y  = floor((ypts[k]-y[0])/dy);
    const int ind_yp = ind_y+1;
    if ( 0 <= ind_x && ind_xp  < M1 && 0 <= ind_y && ind_yp  < M2 ) {
      compute_G_k_BilinearCUDA(
        k, xpts, ypts, x, y,
        ind_x, ind_xp,
        ind_y, ind_yp,
        F, M2,
        dx, dy,
        G[k]
      );
    }
    else{
      G[k] = fill_value;
    }
  }
}

template <typename scalar_t>
__global__ void bilinear_interpolation_kernel_CUDA_linear_extrap_linear(
  scalar_t * G, const scalar_t * F,
  const scalar_t * xpts, const scalar_t * ypts,
  const int M1, const int M2, const int N,
  const double dx, const double dy,
  const scalar_t * x, const scalar_t * y) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    int ind_x  = floor((xpts[k]-x[0])/dx);
    int ind_xp = ind_x+1;

    int ind_y  = floor((ypts[k]-y[0])/dy);
    int ind_yp = ind_y+1;

    if (ind_x<0) {
      ind_x=0;
      ind_xp=1;
    }
    if (ind_xp>=M1) {
      ind_x=M1-2;
      ind_xp=M1-1;
    }
    if (ind_y<0) {
      ind_y=0;
      ind_yp=1;
    }
    if (ind_yp>=M2) {
      ind_y=M2-2;
      ind_yp=M2-1;
    }
      compute_G_k_BilinearCUDA(
        k, xpts, ypts, x, y,
        ind_x, ind_xp,
        ind_y, ind_yp,
        F, M2,
        dx, dy,
        G[k]
      );

    }
  }

template <typename scalar_t>
__global__ void bilinear_interpolation_kernel_CUDA_linear_extrap_nearest(
  scalar_t * G, const scalar_t * F,
  const scalar_t * xpts, const scalar_t * ypts,
  const int M1, const int M2, const int N,
  const double dx, const double dy,
  const scalar_t * x, const scalar_t * y) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    int ind_x  = floor((xpts[k]-x[0])/dx);
    int ind_y  = floor((ypts[k]-y[0])/dy);
    if ( 0 <= ind_x && ind_x  < M1-1 && 0 <= ind_y && ind_y  < M2-1 ) {
      int ind_xp = ind_x+1;
      int ind_yp = ind_y+1;
      compute_G_k_BilinearCUDA(
        k, xpts, ypts, x, y,
        ind_x, ind_xp,
        ind_y, ind_yp,
        F, M2,
        dx, dy,
        G[k]
      );

    }
    else {
      if (ind_x<0) {
        ind_x=0;
      }
      if (ind_x>=M1) {
        ind_x=M1-1;
      }
      if (ind_y<0) {
        ind_y=0;
      }
      if (ind_y>=M2) {
        ind_y=M2-1;
      }
      G[k]=F[ind_x*M2+ind_y];
    }
  }
}



// =====================================================================
//  3-D  trilinear  CUDA kernels
// =====================================================================

template <typename scalar_t>
__device__ void compute_G_k_TrilinearCUDA(
        const int k,
        const scalar_t * xpts, const scalar_t * ypts, const scalar_t * zpts,
        const scalar_t * x, const scalar_t * y, const scalar_t * z,
        const int ind_x, const int ind_xp,
        const int ind_y, const int ind_yp,
        const int ind_z, const int ind_zp,
        const scalar_t * F, const int M2, const int M3,
        const double dx, const double dy, const double dz,
        scalar_t& out
      ) {

  const scalar_t wx0 = x[ind_xp] - xpts[k];
  const scalar_t wx1 = xpts[k]   - x[ind_x];
  const scalar_t wy0 = y[ind_yp] - ypts[k];
  const scalar_t wy1 = ypts[k]   - y[ind_y];
  const scalar_t wz0 = z[ind_zp] - zpts[k];
  const scalar_t wz1 = zpts[k]   - z[ind_z];

  const int s2 = M3;          // stride for y
  const int s1 = M2 * M3;     // stride for x

  out = (
      wx0 * wy0 * wz0 * F[ind_x  * s1 + ind_y  * s2 + ind_z ]
    + wx0 * wy0 * wz1 * F[ind_x  * s1 + ind_y  * s2 + ind_zp]
    + wx0 * wy1 * wz0 * F[ind_x  * s1 + ind_yp * s2 + ind_z ]
    + wx0 * wy1 * wz1 * F[ind_x  * s1 + ind_yp * s2 + ind_zp]
    + wx1 * wy0 * wz0 * F[ind_xp * s1 + ind_y  * s2 + ind_z ]
    + wx1 * wy0 * wz1 * F[ind_xp * s1 + ind_y  * s2 + ind_zp]
    + wx1 * wy1 * wz0 * F[ind_xp * s1 + ind_yp * s2 + ind_z ]
    + wx1 * wy1 * wz1 * F[ind_xp * s1 + ind_yp * s2 + ind_zp]
  ) / (scalar_t)(dx * dy * dz);
}

// --- fill_method = 1: constant padding ---
template <typename scalar_t>
__global__ void trilinear_interpolation_kernel_CUDA_padding(
  scalar_t* G, const scalar_t* F,
  const scalar_t* xpts, const scalar_t* ypts, const scalar_t* zpts,
  const int M1, const int M2, const int M3, const int N,
  const double dx, const double dy, const double dz,
  const scalar_t* x, const scalar_t* y, const scalar_t* z,
  const double fill_value) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    const int ind_x  = floor((xpts[k] - x[0]) / dx);
    const int ind_xp = ind_x + 1;
    const int ind_y  = floor((ypts[k] - y[0]) / dy);
    const int ind_yp = ind_y + 1;
    const int ind_z  = floor((zpts[k] - z[0]) / dz);
    const int ind_zp = ind_z + 1;

    if (0 <= ind_x && ind_xp < M1 &&
        0 <= ind_y && ind_yp < M2 &&
        0 <= ind_z && ind_zp < M3) {
      compute_G_k_TrilinearCUDA(
        k, xpts, ypts, zpts, x, y, z,
        ind_x, ind_xp, ind_y, ind_yp, ind_z, ind_zp,
        F, M2, M3, dx, dy, dz, G[k]);
    } else {
      G[k] = fill_value;
    }
  }
}

// --- fill_method = 2: linear extrapolation (clamp indices) ---
template <typename scalar_t>
__global__ void trilinear_interpolation_kernel_CUDA_linear_extrap_linear(
  scalar_t* G, const scalar_t* F,
  const scalar_t* xpts, const scalar_t* ypts, const scalar_t* zpts,
  const int M1, const int M2, const int M3, const int N,
  const double dx, const double dy, const double dz,
  const scalar_t* x, const scalar_t* y, const scalar_t* z) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    int ind_x  = floor((xpts[k] - x[0]) / dx);
    int ind_xp = ind_x + 1;
    int ind_y  = floor((ypts[k] - y[0]) / dy);
    int ind_yp = ind_y + 1;
    int ind_z  = floor((zpts[k] - z[0]) / dz);
    int ind_zp = ind_z + 1;

    if (ind_x < 0)    { ind_x = 0;    ind_xp = 1; }
    if (ind_xp >= M1)  { ind_x = M1-2; ind_xp = M1-1; }
    if (ind_y < 0)    { ind_y = 0;    ind_yp = 1; }
    if (ind_yp >= M2)  { ind_y = M2-2; ind_yp = M2-1; }
    if (ind_z < 0)    { ind_z = 0;    ind_zp = 1; }
    if (ind_zp >= M3)  { ind_z = M3-2; ind_zp = M3-1; }

    compute_G_k_TrilinearCUDA(
      k, xpts, ypts, zpts, x, y, z,
      ind_x, ind_xp, ind_y, ind_yp, ind_z, ind_zp,
      F, M2, M3, dx, dy, dz, G[k]);
  }
}

// --- fill_method = 3: border / nearest (match grid_sample 'border' mode) ---
//     Clamp query coordinate to grid domain, then interpolate normally.
template <typename scalar_t>
__global__ void trilinear_interpolation_kernel_CUDA_nearest(
  scalar_t* G, const scalar_t* F,
  const scalar_t* xpts, const scalar_t* ypts, const scalar_t* zpts,
  const int M1, const int M2, const int M3, const int N,
  const double dx, const double dy, const double dz,
  const scalar_t* x, const scalar_t* y, const scalar_t* z) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    // Clamp query coords to grid domain
    scalar_t xq = max(x[0], min(xpts[k], x[M1-1]));
    scalar_t yq = max(y[0], min(ypts[k], y[M2-1]));
    scalar_t zq = max(z[0], min(zpts[k], z[M3-1]));

    int ind_x = floor((xq - x[0]) / dx);
    int ind_y = floor((yq - y[0]) / dy);
    int ind_z = floor((zq - z[0]) / dz);

    // Clamp indices to valid interpolation range [0, M-2]
    ind_x = max(0, min(ind_x, M1-2));
    ind_y = max(0, min(ind_y, M2-2));
    ind_z = max(0, min(ind_z, M3-2));

    const int ind_xp = ind_x + 1;
    const int ind_yp = ind_y + 1;
    const int ind_zp = ind_z + 1;

    // Trilinear weights using clamped coordinates
    const scalar_t wx0 = x[ind_xp] - xq;
    const scalar_t wx1 = xq        - x[ind_x];
    const scalar_t wy0 = y[ind_yp] - yq;
    const scalar_t wy1 = yq        - y[ind_y];
    const scalar_t wz0 = z[ind_zp] - zq;
    const scalar_t wz1 = zq        - z[ind_z];

    const int s2 = M3;
    const int s1 = M2 * M3;

    G[k] = (
        wx0 * wy0 * wz0 * F[ind_x  * s1 + ind_y  * s2 + ind_z ]
      + wx0 * wy0 * wz1 * F[ind_x  * s1 + ind_y  * s2 + ind_zp]
      + wx0 * wy1 * wz0 * F[ind_x  * s1 + ind_yp * s2 + ind_z ]
      + wx0 * wy1 * wz1 * F[ind_x  * s1 + ind_yp * s2 + ind_zp]
      + wx1 * wy0 * wz0 * F[ind_xp * s1 + ind_y  * s2 + ind_z ]
      + wx1 * wy0 * wz1 * F[ind_xp * s1 + ind_y  * s2 + ind_zp]
      + wx1 * wy1 * wz0 * F[ind_xp * s1 + ind_yp * s2 + ind_z ]
      + wx1 * wy1 * wz1 * F[ind_xp * s1 + ind_yp * s2 + ind_zp]
    ) / (scalar_t)(dx * dy * dz);
  }
}


// =====================================================================
//  Dispatch functions
// =====================================================================

void interp_cuda(
    const at::Tensor& F,
    at::Tensor& G,
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& xpt,
    const at::Tensor& ypt,
    const int64_t M1,
    const int64_t M2,
    const double dx,
    const double dy,
    const int64_t fill_method,
    const double fill_value,
    const int64_t method
  ) {

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int N  = G.numel();

  // Adaptive block size: use fewer threads per block for small N to
  // reduce wasted lanes and launch overhead.  Always a multiple of the
  // warp size (32) so no partial warps are scheduled.
  const int blockSize = (N <= 128) ? 32 : (N <= 4096) ? 128 : 256;
  const int numBlocks = (N + blockSize - 1) / blockSize;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(F.scalar_type(), "interp_cuda", [&] {

    const scalar_t* F_ptr   = F.contiguous().data_ptr<scalar_t>();
    const scalar_t* xpt_ptr = xpt.contiguous().data_ptr<scalar_t>();
    const scalar_t* ypt_ptr = ypt.contiguous().data_ptr<scalar_t>();
    const scalar_t* x_ptr   = x.contiguous().data_ptr<scalar_t>();
    const scalar_t* y_ptr   = y.contiguous().data_ptr<scalar_t>();
    scalar_t* G_ptr         = G.data_ptr<scalar_t>();

  if (method==0) {
    if(fill_method==1) {
      bilinear_interpolation_kernel_CUDA_padding<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr,
                                      fill_value);
    }
    else if(fill_method==2) {
      bilinear_interpolation_kernel_CUDA_linear_extrap_linear<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr);
    }
    else if(fill_method==3) {
      bilinear_interpolation_kernel_CUDA_linear_extrap_nearest<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr);
    }
  }
  else if(method==1) {
    if(fill_method==1) {
      biquadratic_interpolation_kernel_CUDA_padding<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr,
                                      fill_value);
    }
    else if(fill_method==2) {
      biquadratic_interpolation_kernel_CUDA_linear_extrap_linear<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr);
    }
    else if(fill_method==3) {
      biquadratic_interpolation_kernel_CUDA_linear_extrap_nearest<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr);
    }
  }

  });


}

void interp3d_cuda(
    const at::Tensor& F,
    at::Tensor& G,
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& z,
    const at::Tensor& xpt,
    const at::Tensor& ypt,
    const at::Tensor& zpt,
    const int64_t M1,
    const int64_t M2,
    const int64_t M3,
    const double dx,
    const double dy,
    const double dz,
    const int64_t fill_method,
    const double fill_value
  ) {

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int N = G.numel();

  const int blockSize = (N <= 128) ? 32 : (N <= 4096) ? 128 : 256;
  const int numBlocks = (N + blockSize - 1) / blockSize;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(F.scalar_type(), "interp3d_cuda", [&] {

    const scalar_t* F_ptr   = F.contiguous().data_ptr<scalar_t>();
    const scalar_t* xpt_ptr = xpt.contiguous().data_ptr<scalar_t>();
    const scalar_t* ypt_ptr = ypt.contiguous().data_ptr<scalar_t>();
    const scalar_t* zpt_ptr = zpt.contiguous().data_ptr<scalar_t>();
    const scalar_t* x_ptr   = x.contiguous().data_ptr<scalar_t>();
    const scalar_t* y_ptr   = y.contiguous().data_ptr<scalar_t>();
    const scalar_t* z_ptr   = z.contiguous().data_ptr<scalar_t>();
    scalar_t* G_ptr         = G.data_ptr<scalar_t>();

    if (fill_method == 1) {
      trilinear_interpolation_kernel_CUDA_padding<scalar_t>
        <<<numBlocks, blockSize, 0, stream>>>(
          G_ptr, F_ptr, xpt_ptr, ypt_ptr, zpt_ptr,
          M1, M2, M3, N, dx, dy, dz,
          x_ptr, y_ptr, z_ptr, fill_value);
    }
    else if (fill_method == 2) {
      trilinear_interpolation_kernel_CUDA_linear_extrap_linear<scalar_t>
        <<<numBlocks, blockSize, 0, stream>>>(
          G_ptr, F_ptr, xpt_ptr, ypt_ptr, zpt_ptr,
          M1, M2, M3, N, dx, dy, dz,
          x_ptr, y_ptr, z_ptr);
    }
    else if (fill_method == 3) {
      trilinear_interpolation_kernel_CUDA_nearest<scalar_t>
        <<<numBlocks, blockSize, 0, stream>>>(
          G_ptr, F_ptr, xpt_ptr, ypt_ptr, zpt_ptr,
          M1, M2, M3, N, dx, dy, dz,
          x_ptr, y_ptr, z_ptr);
    }
  });
}

  // Registers CUDA implementations
TORCH_LIBRARY_IMPL(extension_interp, CUDA, m) {
  m.impl("bilinear_interp", &interp_cuda);
  m.impl("trilinear_interp_3d", &interp3d_cuda);
}

}
