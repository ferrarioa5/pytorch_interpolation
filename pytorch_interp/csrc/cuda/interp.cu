
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_interp {

__global__ void bilinear_interpolation_kernel_CUDA(
  float * G, float * F, 
  float * xpts, float * ypts, 
  const int M1, const int M2, const int N,
  const float dx, const float dy, 
  const float * x, const float * y) {

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    const int    ind_x  = floor((xpts[k]-x[0])/dx);
    const int    ind_xp = ind_x+1;

    const int    ind_y  = floor((ypts[k]-y[0])/dy);
    const int    ind_yp = ind_y+1;

    if ( 0 <= ind_x && ind_xp  < M1 && 0 <= ind_y && ind_yp  < M2 ) {
      const float w11 = (x[ind_xp]-xpts[k])*(y[ind_yp]-ypts[k]);
      const float w12 = (x[ind_xp]-xpts[k])*(ypts[k]-y[ind_y]);
      const float w21 = (xpts[k]-x[ind_x])*(y[ind_yp]-ypts[k]);
      const float w22 = (xpts[k]-x[ind_x])*(ypts[k]-y[ind_y]);
      const float numerator = w11*F[ind_x*M2+ind_y] + w12*F[ind_x*M2+ind_y+1] + w21*F[(ind_x+1)*M2+ind_y] + w22*F[(ind_x+1)*M2+ind_y+1];
      const float denominator = (x[ind_xp]-x[ind_x])*(y[ind_yp]-y[ind_y]);
      G[k] = numerator/denominator;
    }
    else{
      G[k] = 0.f;
    }

  }
}

void bilinear_interp_cuda(
    const at::Tensor& F, 
    at::Tensor& G, 
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& xpt,
    const at::Tensor& ypt,
    const int64_t M1, 
    const int64_t M2,
    const double dx, 
    const double dy
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

  bilinear_interpolation_kernel_CUDA<<<(N+255)/256, 256>>>(
                                    G_ptr, F_ptr,
                                    xpt_ptr, ypt_ptr,
                                    M1, M2, N,
                                    dx, dy,
                                    x_ptr, y_ptr);
}

// Registers CUDA implementation
TORCH_LIBRARY_IMPL(extension_interp, CUDA, m) {
  m.impl("bilinear_interp", &bilinear_interp_cuda);
}

}
