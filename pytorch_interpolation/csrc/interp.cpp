
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace extension_interp {

template <typename scalar_t>
scalar_t compute_G_k_Bilinear( const int k,
        const scalar_t * xpts, const scalar_t * ypts,
        const scalar_t * x, const scalar_t * y,
        const int ind_x, const int ind_xp,
        const int ind_y, const int ind_yp,
        const scalar_t * F, const int M2,
        const double dx, const double dy
      ) {

  const scalar_t w11 = (x[ind_xp]-xpts[k])*(y[ind_yp]-ypts[k]);
  const scalar_t w12 = (x[ind_xp]-xpts[k])*(ypts[k]-y[ind_y]);
  const scalar_t w21 = (xpts[k]-x[ind_x])*(y[ind_yp]-ypts[k]);
  const scalar_t w22 = (xpts[k]-x[ind_x])*(ypts[k]-y[ind_y]);

  return (w11*F[ind_x*M2+ind_y] + w12*F[ind_x*M2+ind_yp] + w21*F[ind_xp*M2+ind_y] + w22*F[ind_xp*M2+ind_yp])/(dx*dy);

}


template <typename scalar_t>
scalar_t compute_G_k_BIquadratic( const int k,
        const scalar_t * xpts, const scalar_t * ypts,
        const scalar_t * x, const scalar_t * y,
        const int ind_x, const int ind_xm, const int ind_xp,
        const int ind_y, const int ind_ym, const int ind_yp,
        const scalar_t * F, const int M2,
        const double dx, const double dy
      ) {

  if(ind_xm<0 || ind_ym<0){
    return compute_G_k_Bilinear(
        k, xpts, ypts, x, y,
        ind_x, ind_xp,
        ind_y, ind_yp,
        F, M2,
        dx, dy
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

    // const scalar_t L0_x = (xpts[k]-x[ind_x])*(xpts[k]-x[ind_xp])/( (x[ind_xm]-x[ind_x])*(x[ind_xm]-x[ind_xp]) ); // 2*dx*dx
    // const scalar_t L1_x = (xpts[k]-x[ind_xm])*(xpts[k]-x[ind_xp])/( (x[ind_x]-x[ind_xm])*(x[ind_x]-x[ind_xp]) ); // -dx*dx
    // const scalar_t L2_x = (xpts[k]-x[ind_xm])*(xpts[k]-x[ind_x])/( (x[ind_xp]-x[ind_xm])*(x[ind_xp]-x[ind_x]) ); // 2*dx*dx

    // const scalar_t L0_y = (ypts[k]-y[ind_y])*(ypts[k]-y[ind_yp])/( (y[ind_ym]-y[ind_y])*(y[ind_ym]-y[ind_yp]) );
    // const scalar_t L1_y = (ypts[k]-y[ind_ym])*(ypts[k]-y[ind_yp])/( (y[ind_y]-y[ind_ym])*(y[ind_y]-y[ind_yp]) );
    // const scalar_t L2_y = (ypts[k]-y[ind_ym])*(ypts[k]-y[ind_y])/( (y[ind_yp]-y[ind_ym])*(y[ind_yp]-y[ind_y]) );

    return (L0_x*L0_y*F[ind_xm*M2+ind_ym] + L0_x*L1_y*F[ind_xm*M2+ind_y] + L0_x*L2_y*F[ind_xm*M2+ind_yp]
          + L1_x*L0_y*F[ind_x*M2+ind_ym]   + L1_x*L1_y*F[ind_x*M2+ind_y]   + L1_x*L2_y*F[ind_x*M2+ind_yp]
          + L2_x*L0_y*F[ind_xp*M2+ind_ym] + L2_x*L1_y*F[ind_xp*M2+ind_y] + L2_x*L2_y*F[ind_xp*M2+ind_yp]);

  }

}


template <typename scalar_t>
void biquadratic_interpolation_kernel_CPU_padding(scalar_t * G, scalar_t * F,
                                        const scalar_t * xpts, const scalar_t * ypts,
                                        const int M1, const int M2, const int N,
                                        double dx, double dy,
                                        const scalar_t * x, const scalar_t * y,
                                        double fill_value)
{
  #pragma omp parallel for
  for(int k=0; k<N; k++){
    const int ind_x  = floor((xpts[k]-x[0])/dx);
    const int ind_xm = ind_x-1;
    const int ind_xp = ind_x+1;
    const int ind_y  = floor((ypts[k]-y[0])/dy);
    const int ind_ym = ind_y-1;
    const int ind_yp = ind_y+1;

    if ( 0 <= ind_x && ind_xp  < M1 && 0 <= ind_y && ind_yp < M2 ) {
      G[k] = compute_G_k_BIquadratic(
        k, xpts, ypts, x, y,
        ind_x, ind_xm,  ind_xp,
        ind_y, ind_ym,  ind_yp,
        F, M2,
        dx, dy
      );
    }
    else{
      G[k] = fill_value;
    }
  }
}


template <typename scalar_t>
void biquadratic_interpolation_kernel_CPU_linear_extrap_linear(scalar_t * G, scalar_t * F,
                                        const scalar_t * xpts, const scalar_t * ypts,
                                        const int M1, const int M2, const int N,
                                        double dx, double dy,
                                        const scalar_t * x, const scalar_t * y)
{
  #pragma omp parallel for
  for(int k=0; k<N; k++){

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

    G[k] = compute_G_k_BIquadratic(
      k, xpts, ypts, x, y,
      ind_x, ind_xm,  ind_xp,
      ind_y, ind_ym,  ind_yp,
      F, M2,
      dx, dy
    );

  }
}


template <typename scalar_t>
void biquadratic_interpolation_kernel_CPU_nearest(scalar_t * G, scalar_t * F,
                                        const scalar_t * xpts, const scalar_t * ypts,
                                        const int M1, const int M2, const int N,
                                        double dx, double dy,
                                        const scalar_t * x, const scalar_t * y)
{

  #pragma omp parallel for
  for(int k=0; k<N; k++){
    int ind_x  = floor((xpts[k]-x[0])/dx);
    int ind_y  = floor((ypts[k]-y[0])/dy);
    int ind_xm = ind_x-1;
    int ind_xp = ind_x+1;
    int ind_ym = ind_y-1;
    int ind_yp = ind_y+1;

    if ( 0 <= ind_xm && ind_xp  < M1 && 0 <= ind_ym && ind_yp  < M2 ) {

      G[k] = compute_G_k_BIquadratic(
        k, xpts, ypts, x, y,
        ind_x, ind_xm,  ind_xp,
        ind_y, ind_ym,  ind_yp,
        F, M2,
        dx, dy
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
void bilinear_interpolation_kernel_CPU_padding(scalar_t * G, scalar_t * F,
                                        const scalar_t * xpts, const scalar_t * ypts,
                                        const int M1, const int M2, const int N,
                                        double dx, double dy,
                                        const scalar_t * x, const scalar_t * y,
                                        double fill_value)
{
  #pragma omp parallel for
  for(int k=0; k<N; k++){

    const int ind_x  = floor((xpts[k]-x[0])/dx);
    const int ind_xp = ind_x+1;
    const int ind_y  = floor((ypts[k]-y[0])/dy);
    const int ind_yp = ind_y+1;

    if ( 0 <= ind_x && ind_xp  < M1 && 0 <= ind_y && ind_yp  < M2 ) {
      G[k] = compute_G_k_Bilinear(
        k, xpts, ypts, x, y,
        ind_x, ind_xp,
        ind_y, ind_yp,
        F, M2,
        dx, dy
      );
    }
    else{
      G[k] = fill_value;
    }
  }
}


template <typename scalar_t>
void bilinear_interpolation_kernel_CPU_linear_extrap_linear(scalar_t * G, scalar_t * F,
                                        const scalar_t * xpts, const scalar_t * ypts,
                                        const int M1, const int M2, const int N,
                                        double dx, double dy,
                                        const scalar_t * x, const scalar_t * y)
{

  #pragma omp parallel for
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
    G[k] = compute_G_k_Bilinear(
      k, xpts, ypts, x, y,
      ind_x, ind_xp,
      ind_y, ind_yp,
      F, M2,
      dx, dy
    );
  }
}


template <typename scalar_t>
void bilinear_interpolation_kernel_CPU_linear_extrap_nearest(scalar_t * G, scalar_t * F,
  const scalar_t * xpts, const scalar_t * ypts,
  const int M1, const int M2, const int N,
  double dx, double dy,
  const scalar_t * x, const scalar_t * y) {

  #pragma omp parallel for
  for(int k=0; k<N; k++){
    int ind_x = floor((xpts[k]-x[0])/dx);
    int ind_y = floor((ypts[k]-y[0])/dy);

    if ( 0 <= ind_x && ind_x  < M1-1 && 0 <= ind_y && ind_y  < M2-1 ) {
      int ind_xp = ind_x+1;
      int ind_yp = ind_y+1;
      G[k] = compute_G_k_Bilinear(
        k, xpts, ypts, x, y,
        ind_x, ind_xp,
        ind_y, ind_yp,
        F, M2,
        dx, dy
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


void interp_cpu(
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(F.scalar_type(), "interp_cpu", [&] {

    at::Tensor F_contig = F.contiguous();
    scalar_t* F_ptr = F_contig.data_ptr<scalar_t>();

    at::Tensor xpt_contig = xpt.contiguous();
    const scalar_t* xpt_ptr = xpt_contig.data_ptr<scalar_t>();

    at::Tensor ypt_contig = ypt.contiguous();
    const scalar_t* ypt_ptr = ypt_contig.data_ptr<scalar_t>();

    at::Tensor x_contig = x.contiguous();
    const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();

    at::Tensor y_contig = y.contiguous();
    const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();

    scalar_t* G_ptr = G.data_ptr<scalar_t>();
    const int N  = G.numel();

    // bilinear_interpolation_kernel_CPU_padding(G_ptr, F_ptr,
    //                             xpt_ptr, ypt_ptr,
    //                             M1, M2, N,
    //                             dx, dy,
    //                             x_ptr, y_ptr,
    //                             fill_value);

  if (method==0) {
    if(fill_method==1) {
      bilinear_interpolation_kernel_CPU_padding(G_ptr, F_ptr,
                                        xpt_ptr, ypt_ptr,
                                        M1, M2, N,
                                        dx, dy,
                                        x_ptr, y_ptr,
                                        fill_value);
    }
    else if(fill_method==2) {
      bilinear_interpolation_kernel_CPU_linear_extrap_linear(G_ptr, F_ptr,
                                          xpt_ptr, ypt_ptr,
                                          M1, M2, N,
                                          dx, dy,
                                          x_ptr, y_ptr);
    }
    else if(fill_method==3) {
      bilinear_interpolation_kernel_CPU_linear_extrap_nearest(G_ptr, F_ptr,
                                        xpt_ptr, ypt_ptr,
                                        M1, M2, N,
                                        dx, dy,
                                        x_ptr, y_ptr);
    }
  }
  else if(method==1) {
    if(fill_method==1) {
      biquadratic_interpolation_kernel_CPU_padding(G_ptr, F_ptr,
                                        xpt_ptr, ypt_ptr,
                                        M1, M2, N,
                                        dx, dy,
                                        x_ptr, y_ptr,
                                        fill_value);
    }
    else if(fill_method==2) {
      biquadratic_interpolation_kernel_CPU_linear_extrap_linear(G_ptr, F_ptr,
                                          xpt_ptr, ypt_ptr,
                                          M1, M2, N,
                                          dx, dy,
                                          x_ptr, y_ptr);
    }
    else if(fill_method==3) {
      biquadratic_interpolation_kernel_CPU_nearest(G_ptr, F_ptr,
                                        xpt_ptr, ypt_ptr,
                                        M1, M2, N,
                                        dx, dy,
                                        x_ptr, y_ptr);
    }
  }



  });



}

// Defines the operators
TORCH_LIBRARY(extension_interp, m) {
  m.def("bilinear_interp(Tensor F, Tensor G , Tensor x, Tensor y, Tensor xpt, Tensor ypt, int M1, int M2, float dx, float dy, int fill_method, float fill_value, int method) -> ()");
}

// Registers CPU implementation
TORCH_LIBRARY_IMPL(extension_interp, CPU, m) {
  m.impl("bilinear_interp", &interp_cpu);
}

}
