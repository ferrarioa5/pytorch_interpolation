
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void compute_G_k_BilinearCUDA( const int k,
        const float * xpts, const float * ypts,
        const float * x, const float * y,
        const int ind_x, const int ind_xp,
        const int ind_y, const int ind_yp,
        const float * F, const int M2,
        const float dx, const float dy,
        float& out
      ) {

  const float w11 = (x[ind_xp]-xpts[k])*(y[ind_yp]-ypts[k]);
  const float w12 = (x[ind_xp]-xpts[k])*(ypts[k]-y[ind_y]);
  const float w21 = (xpts[k]-x[ind_x])*(y[ind_yp]-ypts[k]);
  const float w22 = (xpts[k]-x[ind_x])*(ypts[k]-y[ind_y]);

  out = (w11*F[ind_x*M2+ind_y] + w12*F[ind_x*M2+ind_yp] + w21*F[ind_xp*M2+ind_y] + w22*F[ind_xp*M2+ind_yp])/(dx*dy);

}

__device__ void compute_G_k_BIquadraticCUDA( const int k,
        const float * xpts, const float * ypts,
        const float * x, const float * y,
        const int ind_x, const int ind_xm, const int ind_xp,
        const int ind_y, const int ind_ym, const int ind_yp,
        const float * F, const int M2,
        const float dx, const float dy,
        float& out
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

    const float L0_x = (xpts[k]-x[ind_x])*(xpts[k]-x[ind_xp])/( 2*dx*dx );
    const float L1_x = (xpts[k]-x[ind_xm])*(xpts[k]-x[ind_xp])/( -dx*dx );
    const float L2_x = (xpts[k]-x[ind_xm])*(xpts[k]-x[ind_x])/( 2*dx*dx );

    const float L0_y = (ypts[k]-y[ind_y])*(ypts[k]-y[ind_yp])/( 2*dy*dy );
    const float L1_y = (ypts[k]-y[ind_ym])*(ypts[k]-y[ind_yp])/( -dy*dy );
    const float L2_y = (ypts[k]-y[ind_ym])*(ypts[k]-y[ind_y])/( 2*dy*dy );

    out = (L0_x*L0_y*F[ind_xm*M2+ind_ym] + L0_x*L1_y*F[ind_xm*M2+ind_y] + L0_x*L2_y*F[ind_xm*M2+ind_yp]
          + L1_x*L0_y*F[ind_x*M2+ind_ym]   + L1_x*L1_y*F[ind_x*M2+ind_y]   + L1_x*L2_y*F[ind_x*M2+ind_yp]
          + L2_x*L0_y*F[ind_xp*M2+ind_ym] + L2_x*L1_y*F[ind_xp*M2+ind_y] + L2_x*L2_y*F[ind_xp*M2+ind_yp]);

  }

}

namespace extension_interp {

__global__ void biquadratic_interpolation_kernel_CUDA_padding(
  float * G, float * F,
  float * xpts, float * ypts,
  const int M1, const int M2, const int N,
  const float dx, const float dy,
  const float * x, const float * y,
  const float fill_value) {

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

__global__ void biquadratic_interpolation_kernel_CUDA_linear_extrap_linear(
  float * G, float * F,
  float * xpts, float * ypts,
  const int M1, const int M2, const int N,
  const float dx, const float dy,
  const float * x, const float * y) {

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

__global__ void biquadratic_interpolation_kernel_CUDA_linear_extrap_nearest(
  float * G, float * F,
  float * xpts, float * ypts,
  const int M1, const int M2, const int N,
  const float dx, const float dy,
  const float * x, const float * y) {

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


__global__ void bilinear_interpolation_kernel_CUDA_padding(
  float * G, float * F,
  float * xpts, float * ypts,
  const int M1, const int M2, const int N,
  const float dx, const float dy,
  const float * x, const float * y,
  const float fill_value) {

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

__global__ void bilinear_interpolation_kernel_CUDA_linear_extrap_linear(
  float * G, float * F,
  float * xpts, float * ypts,
  const int M1, const int M2, const int N,
  const float dx, const float dy,
  const float * x, const float * y) {

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

__global__ void bilinear_interpolation_kernel_CUDA_linear_extrap_nearest(
  float * G, float * F,
  float * xpts, float * ypts,
  const int M1, const int M2, const int N,
  const float dx, const float dy,
  const float * x, const float * y) {

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

  at::Tensor F_contig = F.contiguous();
  float* F_ptr = F_contig.data_ptr<float>();

  at::Tensor xpt_contig = xpt.contiguous();
  float* xpt_ptr = xpt_contig.data_ptr<float>();

  at::Tensor ypt_contig = ypt.contiguous();
  float* ypt_ptr = ypt_contig.data_ptr<float>();

  at::Tensor x_contig = x.contiguous();
  const float* x_ptr = x_contig.data_ptr<float>();

  at::Tensor y_contig = y.contiguous();
  const float* y_ptr = y_contig.data_ptr<float>();

  const int N  = xpt_contig.numel();

  float* G_ptr = G.data_ptr<float>();

  if (method==0) {
    if(fill_method==1) {
      bilinear_interpolation_kernel_CUDA_padding<<<(N+255)/256, 256>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr,
                                      fill_value);
    }
    else if(fill_method==2) {
      bilinear_interpolation_kernel_CUDA_linear_extrap_linear<<<(N+255)/256, 256>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr);
    }
    else if(fill_method==3) {
      bilinear_interpolation_kernel_CUDA_linear_extrap_nearest<<<(N+255)/256, 256>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr);
    }
  }
  else if(method==1) {
    if(fill_method==1) {
      biquadratic_interpolation_kernel_CUDA_padding<<<(N+255)/256, 256>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr,
                                      fill_value);
    }
    else if(fill_method==2) {
      biquadratic_interpolation_kernel_CUDA_linear_extrap_linear<<<(N+255)/256, 256>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr);
    }
    else if(fill_method==3) {
      biquadratic_interpolation_kernel_CUDA_linear_extrap_nearest<<<(N+255)/256, 256>>>(
                                      G_ptr, F_ptr,
                                      xpt_ptr, ypt_ptr,
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr);
    }

  }
}

  // Registers CUDA implementation
TORCH_LIBRARY_IMPL(extension_interp, CUDA, m) {
  m.impl("bilinear_interp", &interp_cuda);
}

}
