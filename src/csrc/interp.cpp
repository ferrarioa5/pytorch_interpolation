
#include <torch/extension.h>
#include <vector>

namespace extension_interp {
void bilinear_interpolation_kernel_CPU(float * __restrict__ G, const float * __restrict__ F, 
                                                  const float * __restrict__ xpts, const float * __restrict__ ypts, 
                                                  const int M1, const int M2, const int N,
                                                  const float dx, const float dy, 
                                                  const float xmin, const float ymin,
                                                  const float* x, const float* y)
{

  for(int k=0; k<N; k++){

    const int    ind_x = floor((xpts[k]-xmin)/dx);
    const float  a     = xpts[k]-x[ind_x];
    const float am     = 1-a;

    const int    ind_y = floor((ypts[k]-ymin)/dy);
    const float  b     = ypts[k]-y[ind_y];
    const float bm     = 1-b;

    float d00, d01, d10, d11;
    if ((0 < (ind_x)   < M1)&&(0 < (ind_y)   < M2))  d00 = F[ind_x*M2+ind_y];       else    d00 = 0.f;
    if ((0 < (ind_x+1) < M1)&&(0 < (ind_y)   < M2))  d10 = F[ind_x*M2+ind_y+1];     else    d10 = 0.f;
    if ((0 < (ind_x)   < M1)&&(0 < (ind_y+1) < M2))  d01 = F[(ind_x+1)*M2+ind_y];   else    d01 = 0.f;
    if ((0 < (ind_x+1) < M1)&&(0 < (ind_y+1) < M2))  d11 = F[(ind_x+1)*M2+ind_y+1]; else    d11 = 0.f;


    G[k] = a*bm*d10 + am*bm*d00 + a*b*d11 + b*am*d01;

  }

}


torch::Tensor bilinear_interp_cpu(
    const at::Tensor& F, 
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& xpt,
    const at::Tensor& ypt
  ) {

  // a bunch of these operations could be improved by having an
  // interpolation class with a dedicated call to the interpolation
  // on the query points

  TORCH_CHECK(F.dtype() == at::kFloat);
  TORCH_CHECK(x.dtype() == at::kFloat);
  TORCH_CHECK(y.dtype() == at::kFloat);
  TORCH_CHECK(xpt.dtype() == at::kFloat);
  TORCH_CHECK(ypt.dtype() == at::kFloat);

  TORCH_CHECK(xpt.sizes() == ypt.sizes());

  TORCH_INTERNAL_ASSERT(F.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(xpt.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(ypt.device().type() == at::DeviceType::CPU);

  const int N  = xpt.sizes()[0];
  const int M1 = F.sizes()[0];
  const int M2 = F.sizes()[1];

  const float xmin = torch::min(x).item<float>();
  const float ymin = torch::min(y).item<float>();
  const float dx = x[1].item<float>()-x[0].item<float>();
  const float dy = y[1].item<float>()-y[0].item<float>();

  torch::Tensor G = torch::zeros({N});

  at::Tensor F_contig = F.contiguous();
  const float* F_ptr = F_contig.data_ptr<float>();

  at::Tensor xpt_contig = xpt.contiguous();
  const float* xpt_ptr = xpt_contig.data_ptr<float>();

  at::Tensor ypt_contig = ypt.contiguous();
  const float* ypt_ptr = ypt_contig.data_ptr<float>();

  at::Tensor x_contig = x.contiguous();
  const float* x_ptr = x_contig.data_ptr<float>();

  at::Tensor y_contig = y.contiguous();
  const float* y_ptr = y_contig.data_ptr<float>();

  float* G_ptr = G.data_ptr<float>();

  bilinear_interpolation_kernel_CPU(G_ptr, F_ptr, 
                                    xpt_ptr, ypt_ptr, 
                                    M1, M2, N,
                                    dx, dy,
                                    xmin, ymin,
                                    x_ptr, y_ptr);
  return G;

}


// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(extension_interp, m) {
  m.def("bilinear_interp(Tensor F, Tensor x, Tensor y, Tensor xpt, Tensor ypt) -> Tensor");
}

// Registers CPU implementation
TORCH_LIBRARY_IMPL(extension_interp, CPU, m) {
  m.impl("bilinear_interp", &bilinear_interp_cpu);
}

}
