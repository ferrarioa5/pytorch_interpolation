
#include <torch/extension.h>
#include <vector>

namespace extension_interp {
void bilinear_interpolation_kernel_CPU_zero_padding(float * G, float * F, 
                                        const float * xpts, const float * ypts, 
                                        const int M1, const int M2, const int N,
                                        const float dx, const float dy, 
                                        const float * x, const float * y,
                                        const float fill_value)
{

  for(int k=0; k<N; k++){
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
      G[k] = fill_value;
    }
  }
}

void bilinear_interpolation_kernel_CPU_linear_extrap(float * G, float * F, 
                                        const float * xpts, const float * ypts, 
                                        const int M1, const int M2, const int N,
                                        const float dx, const float dy, 
                                        const float * x, const float * y)
{

  for(int k=0; k<N; k++){
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
    const float w11 = (x[ind_xp]-xpts[k])*(y[ind_yp]-ypts[k]);
    const float w12 = (x[ind_xp]-xpts[k])*(ypts[k]-y[ind_y]);
    const float w21 = (xpts[k]-x[ind_x])*(y[ind_yp]-ypts[k]);
    const float w22 = (xpts[k]-x[ind_x])*(ypts[k]-y[ind_y]);
    const float numerator = w11*F[ind_x*M2+ind_y] + w12*F[ind_x*M2+ind_y+1] + w21*F[(ind_x+1)*M2+ind_y] + w22*F[(ind_x+1)*M2+ind_y+1];
    const float denominator = (x[ind_xp]-x[ind_x])*(y[ind_yp]-y[ind_y]);
    G[k] = numerator/denominator;
  }
}



void bilinear_interp_cpu(
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
    const double fill_value
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

  float* G_ptr = G.data_ptr<float>();
  const int N  = G.numel();

  if(fill_method==1) {
    bilinear_interpolation_kernel_CPU_zero_padding(G_ptr, F_ptr, 
                                      xpt_ptr, ypt_ptr, 
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr,
                                      fill_value);
  }
  else if(fill_method==2) {
   bilinear_interpolation_kernel_CPU_linear_extrap(G_ptr, F_ptr, 
                                      xpt_ptr, ypt_ptr, 
                                      M1, M2, N,
                                      dx, dy,
                                      x_ptr, y_ptr);
  }
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(extension_interp, m) {
  m.def("bilinear_interp(Tensor F, Tensor G , Tensor x, Tensor y, Tensor xpt, Tensor ypt, int M1, int M2, float dx, float dy, int fill_method, float fill_value) -> ()");
}

// Registers CPU implementation
TORCH_LIBRARY_IMPL(extension_interp, CPU, m) {
  m.impl("bilinear_interp", &bilinear_interp_cpu);
}

}
