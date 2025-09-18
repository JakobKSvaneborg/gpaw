#include "../gpu.h"
#include "../gpu-complex.h"
#include "numpy/arrayobject.h"
#include <cassert>

#include "../cpp/gpu_core.hpp"
#include "../../gpaw_utils.h"

#define BETA   0.066725
#define GAMMA  0.031091
#define MU     0.2195164512208958 // PBE mod in libxc
//#define MU     0.2195149727645171 from libxc
#define C2     0.26053088059892404
#define C0I    0.238732414637843
#define C1    -0.45816529328314287
#define CC1    1.9236610509315362
#define CC2    2.5648814012420482
#define IF2    0.58482236226346462
#define C3     0.10231023756535741
#define C0     4.1887902047863905
#define THIRD  0.33333333333333333
#define NMIN   1.0E-10

template <typename Tcomplex, typename Treal>
__global__ void calculate_residual_kernel(int nG, int nn,
					       				  Tcomplex* residual_nG,
										  Treal* eps_n,
										  Tcomplex* wf_nG)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    int g = threadIdx.y + blockIdx.y * blockDim.y;
    if ((g < nG) && (n < nn))
    {
		residual_nG[n*nG + g] = residual_nG[n*nG + g] - wf_nG[n*nG + g] * eps_n[n];
    }
}

// This is the [i,j,0] slice of contiguous array
#define MAT(array, nx, ny, nz, b, i, j) (array[(b) * (nx) * (ny) * (nz) + (i) * (ny) * (nz) + (j) * (nz)])

template <typename Tcomplex>
__global__ void pw_amend_insert_realwf(int nb, int nx, int ny, int nz, int n, int m, Tcomplex* array_nQ)
{
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    if (b < nb)
    {
        // t[0, -m:] = t[0, m:0:-1].conj()
        if (i < m)
        {
            Tcomplex value = MAT(array_nQ, nx, ny, nz, b, 0, m - i);
            value.y = -value.y;
            MAT(array_nQ, nx, ny, nz, b, 0, ny - m + i) = value;
        }

        if (i < n)
        {
            for (int j=0; j<m; j++)
            {
                // t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
                Tcomplex value = MAT(array_nQ, nx, ny, nz, b, nx - n + i, m - j);
                value.y = -value.y;
                MAT(array_nQ, nx, ny, nz, b, n - i, ny - m + j) = value;

                // t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
                value = MAT(array_nQ, nx, ny, nz, b, n - i, m - j);
                value.y = -value.y;
                MAT(array_nQ, nx, ny, nz, b, nx - n + i, ny - m + j) = value;
            }
            Tcomplex value = MAT(array_nQ, nx, ny, nz, b, n - i, 0);
            value.y = -value.y;
            MAT(array_nQ, nx, ny, nz, b, nx - n + i, 0) = value;
            }
        }
}


CLINKAGE void calculate_residual_launch_kernel(
				      int dtypenum,
					  int nG,
				      int nn,
				      void* residual_nG,
				      void* eps_n,
				      void* wf_nG,
					  gpuStream_t stream)
{
    if ((nG == 0) || (nn == 0))
    {
        return;
    }

	const dim3 blocks((nn+15)/16, (nG+15)/16);
	const dim3 threads(16, 16);
	const int shmem = 0;

    if (dtypenum==NP_DOUBLE_COMPLEX)
    {
		gpaw::launch_kernel(
			calculate_residual_kernel<gpuDoubleComplex, double>,
			blocks,
			threads,
			shmem,
			stream,
			nG,
			nn,
			static_cast<gpuDoubleComplex*>(residual_nG),
			static_cast<double*>(eps_n),
			static_cast<gpuDoubleComplex*>(wf_nG)
		);
    }
    else if (dtypenum==NP_FLOAT_COMPLEX)
    {
		gpaw::launch_kernel(
			calculate_residual_kernel<gpuFloatComplex, float>,
			blocks,
			threads,
			shmem,
			stream,
			nG,
			nn,
			static_cast<gpuFloatComplex*>(residual_nG),
			static_cast<float*>(eps_n),
			static_cast<gpuFloatComplex*>(wf_nG)
		);
    }
	else if (dtypenum==NP_FLOAT)
	{
		gpaw::launch_kernel(
			calculate_residual_kernel<float, float>,
			blocks,
			threads,
			shmem,
			stream,
			nG,
			nn,
			static_cast<float*>(residual_nG),
			static_cast<float*>(eps_n),
			static_cast<float*>(wf_nG)
		);
	}
	else if (dtypenum==NP_DOUBLE)
	{
		gpaw::launch_kernel(
			calculate_residual_kernel<double, double>,
			blocks,
			threads,
			shmem,
			stream,
			nG,
			nn,
			static_cast<double*>(residual_nG),
			static_cast<double*>(eps_n),
			static_cast<double*>(wf_nG)
		);
	}
	else
	{
		assert(false);
	}
}


template <bool gga> __device__ double pbe_exchange(double n, double rs, double a2,
						   double* dedrs, double* deda2)
{
    double e = C1 / rs;
    *dedrs = -e / rs;
    if (gga)
    {
	double kappa = 0.804;
	double c = C2 * rs / n;
	c *= c;
	double s2 = a2 * c;
	double x = 1.0 + MU * s2 / kappa;
	double Fx = 1.0 + kappa - kappa / x;
	double dFxds2 = MU / (x * x);
	double ds2drs = 8.0 * c * a2 / rs;
	*dedrs = *dedrs * Fx + e * dFxds2 * ds2drs;
	*deda2 = e * dFxds2 * c;
	e *= Fx;
    }
    return e;
}


__device__ double G(double rtrs, double A, double alpha1,
		    double beta1, double beta2, double beta3, double beta4,
		    double* dGdrs)
{
  double Q0 = -2.0 * A * (1.0 + alpha1 * rtrs * rtrs);
  double Q1 = 2.0 * A * rtrs * (beta1 +
				rtrs * (beta2 +
					rtrs * (beta3 +
						rtrs * beta4)));
  double G1 = Q0 * log(1.0 + 1.0 / Q1);
  double dQ1drs = A * (beta1 / rtrs + 2.0 * beta2 +
		       rtrs * (3.0 * beta3 + 4.0 * beta4 * rtrs));
  *dGdrs = -2.0 * A * alpha1 * G1 / Q0 - Q0 * dQ1drs / (Q1 * (Q1 + 1.0));
  return G1;
}


template <bool gga, int nspin> __device__ double pbe_correlation(double n, double rs, double zeta, double a2,
						       double* dedrs, double* dedzeta, double* deda2)
{
  bool spinpol = nspin == 2;
  double rtrs = sqrt(rs);
  double de0drs;
  double e0 = G(rtrs, GAMMA, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294,
		&de0drs);
  double e;
  double xp = 117.0;
  double xm = 117.0;
  if (spinpol)
    {
      double de1drs;
      double e1 = G(rtrs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662,
		    0.62517, &de1drs);
      double dalphadrs;
      double alpha = -G(rtrs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026,
			0.49671, &dalphadrs);
      dalphadrs = -dalphadrs;
      double zp = 1.0 + zeta;
      double zm = 1.0 - zeta;
      xp = pow(zp, THIRD);
      xm = pow(zm, THIRD);
      double f = CC1 * (zp * xp + zm * xm - 2.0);
      double f1 = CC2 * (xp - xm);
      double zeta2 = zeta * zeta;
      double zeta3 = zeta2 * zeta;
      double zeta4 = zeta2 * zeta2;
      double x = 1.0 - zeta4;
      *dedrs = (de0drs * (1.0 - f * zeta4) +
	       de1drs * f * zeta4 +
	       dalphadrs * f * x * IF2);
      *dedzeta = (4.0 * zeta3 * f * (e1 - e0 - alpha * IF2) +
		 f1 * (zeta4 * e1 - zeta4 * e0 + x * alpha * IF2));
      e = e0 + alpha * IF2 * f * x + (e1 - e0) * f * zeta4;
    }
  else
    {
      *dedrs = de0drs;
      e = e0;
    }
  if (gga)
    {
      double n2 = n * n;
      double t2;
      double y;
      double phi = 117.0;
      double phi2 = 117.0;
      double phi3 = 117.0;
      if (spinpol)
	{
	  phi = 0.5 * (xp * xp + xm * xm);
	  phi2 = phi * phi;
	  phi3 = phi * phi2;
	  t2 = C3 * a2 * rs / (n2 * phi2);
	  y = -e / (GAMMA * phi3);
	}
      else
	{
	  t2 = C3 * a2 * rs / n2;
	  y = -e / GAMMA;
	}
      double x = exp(y);
      double A;
      if (x != 1.0)
	A = BETA / (GAMMA * (x - 1.0));
      else
	A = BETA / (GAMMA * y);
      double At2 = A * t2;
      double nom = 1.0 + At2;
      double denom = nom + At2 * At2;
      double H = GAMMA * log( 1.0 + BETA * t2 * nom / (denom * GAMMA));
      double tmp = (GAMMA * BETA /
		    (denom * (BETA * t2 * nom + GAMMA * denom)));
      double tmp2 = A * A * x / BETA;
      double dAdrs = tmp2 * *dedrs;
      if (spinpol)
	{
	  H *= phi3;
	  tmp *= phi3;
	  dAdrs /= phi3;
	}
      double dHdt2 = (1.0 + 2.0 * At2) * tmp;
      double dHdA = -At2 * t2 * t2 * (2.0 + At2) * tmp;
      *dedrs += dHdt2 * 7 * t2 / rs + dHdA * dAdrs;
      *deda2 = dHdt2 * C3 * rs / n2;
      if (spinpol)
	{
	  double dphidzeta = (1.0 / xp - 1.0 / xm) / 3.0;
	  double dAdzeta = tmp2 * (*dedzeta -
				   3.0 * e * dphidzeta / phi) / phi3;
	  *dedzeta += ((3.0 * H / phi - dHdt2 * 2.0 * t2 / phi ) * dphidzeta +
		      dHdA * dAdzeta);
	  *deda2 /= phi2;
	}
      e += H;
    }
  return e;
}


template <int nspin, bool gga> __global__ void evaluate_ldaorgga_kernel(int ng,
									double* n_sg,
									double* v_sg,
									double* e_g,
									double* sigma_xg,
									double* dedsigma_xg)
{
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= ng)
    {
	return;
    }
    if (nspin == 1)
    {
	double n = n_sg[g];
	if (n < NMIN)
	  n = NMIN;
	double rs = pow(C0I / n, THIRD);
	double dexdrs;
	double dexda2;
	double ex;
	double decdrs;
	double decda2;
	double ec;
	if (gga)
	  {
	    double a2 = sigma_xg[g];
	    ex = pbe_exchange<gga>(n, rs, a2, &dexdrs, &dexda2);
	    ec = pbe_correlation<gga, nspin>(n, rs, 0.0, a2, &decdrs, 0, &decda2);
	    dedsigma_xg[g] = n * (dexda2 + decda2);
	  }
	else
	  {
	    ex = pbe_exchange<gga>(n, rs, 0.0, &dexdrs, 0);
	    ec = pbe_correlation<gga, nspin>(n, rs, 0.0, 0.0, &decdrs, 0, 0);
	  }
	e_g[g] = n * (ex + ec);
	v_sg[g] += ex + ec - rs * (dexdrs + decdrs) / 3.0;
    }
    else
    {
	const double* na_g = n_sg;
	double* va_g = v_sg;
	const double* nb_g = na_g + ng;
	double* vb_g = va_g + ng;

	const double* sigma0_g = 0;
	const double* sigma1_g = 0;
	const double* sigma2_g = 0;
	double* dedsigma0_g = 0;
	double* dedsigma1_g = 0;
	double* dedsigma2_g = 0;

	if (gga)
	{
	    sigma0_g = sigma_xg;
	    sigma1_g = sigma0_g + ng;
	    sigma2_g = sigma1_g + ng;
	    dedsigma0_g = dedsigma_xg;
	    dedsigma1_g = dedsigma0_g + ng;
	    dedsigma2_g = dedsigma1_g + ng;
	}

	double na = 2.0 * na_g[g];
	if (na < NMIN)
	  na = NMIN;
	double rsa = pow(C0I / na, THIRD);
	double nb = 2.0 * nb_g[g];
	if (nb < NMIN)
	  nb = NMIN;
	double rsb = pow(C0I / nb, THIRD);
	double n = 0.5 * (na + nb);
	double rs = pow(C0I / n, THIRD);
	double zeta = 0.5 * (na - nb) / n;
	double dexadrs;
	double dexada2;
	double exa;
	double dexbdrs;
	double dexbda2;
	double exb;
	double decdrs;
	double decdzeta;
	double decda2;
	double ec;
	if (gga)
	{
	    exa = pbe_exchange<gga>(na, rsa, 4.0 * sigma0_g[g],
			       &dexadrs, &dexada2);
	    exb = pbe_exchange<gga>(nb, rsb, 4.0 * sigma2_g[g],
				   &dexbdrs, &dexbda2);
	    double a2 = sigma0_g[g] + 2 * sigma1_g[g] + sigma2_g[g];
	    ec = pbe_correlation<gga, nspin>(n, rs, zeta, a2,
					     &decdrs, &decdzeta, &decda2);
	    dedsigma0_g[g] = 2 * na * dexada2 + n * decda2;
	    dedsigma1_g[g] = 2 * n * decda2;
	    dedsigma2_g[g] = 2 * nb * dexbda2 + n * decda2;
	}
	else
	{
	   exa = pbe_exchange<gga>(na, rsa, 0.0, &dexadrs, 0);
	   exb = pbe_exchange<gga>(nb, rsb, 0.0, &dexbdrs, 0);
	   ec = pbe_correlation<gga, nspin>(n, rs, zeta, 0.0,
					    &decdrs, &decdzeta, 0);
	}

	e_g[g] = 0.5 * (na * exa + nb * exb) + n * ec;
	va_g[g] += (exa + ec -
		    (rsa * dexadrs + rs * decdrs) / 3.0 -
		    (zeta - 1.0) * decdzeta);
	vb_g[g] += (exb + ec -
		    (rsb * dexbdrs + rs * decdrs) / 3.0 -
		    (zeta + 1.0) * decdzeta);
    }
}

CLINKAGE void evaluate_pbe_launch_kernel(int nspin, int ng,
				double* n,
				double* v,
				double* e,
				double* sigma,
				double* dedsigma,
				gpuStream_t stream )
{
	const dim3 blocks(dim3((ng+255)/256));
	const dim3 threads(256);
	const int shmem = 0;

    if (nspin == 1)
    {
		gpaw::launch_kernel(
			evaluate_ldaorgga_kernel<1, true>,
			blocks,
			threads,
			shmem,
			stream,
			ng,
			n,
			v,
			e,
			sigma,
			dedsigma
		);
    }
    else if (nspin == 2)
    {
		gpaw::launch_kernel(
			evaluate_ldaorgga_kernel<2, true>,
			blocks,
			threads,
			shmem,
			stream,
			ng,
			n,
			v,
			e,
			sigma,
			dedsigma
		);
    }
}

CLINKAGE void evaluate_lda_launch_kernel(int nspin, int ng,
				double* n,
				double* v,
				double* e,
				gpuStream_t stream)
{
    if (!ng)
    {
        return;
    }
	const dim3 blocks((ng+255)/256);
	const dim3 threads(256);
	const int shmem = 0;

    if (nspin == 1)
    {
		gpaw::launch_kernel(
			evaluate_ldaorgga_kernel<1, false>,
			blocks,
			threads,
			shmem,
			stream,
			ng,
			n,
			v,
			e,
			nullptr,
			nullptr
		);
    }
    else if (nspin == 2)
    {
		gpaw::launch_kernel(
			evaluate_ldaorgga_kernel<2, false>,
			blocks,
			threads,
			shmem,
			stream,
			ng,
			n,
			v,
			e,
			nullptr,
			nullptr
		);
    }
}

template <typename Tcomplex, typename Treal>
__global__ void pw_insert_many(int nb,
				  int nG,
				  int nQ,
				  Tcomplex* c_nG,
				  npy_int32* Q_G,
				  Treal scale,
				  Tcomplex* tmp_nQ)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    int b = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ npy_int32 locQ_G[16];
    if (threadIdx.y == 0)
	locQ_G[threadIdx.x] = Q_G[G];
    __syncthreads();

    if ((G < nG) && (b < nb))
    {
	npy_int32 Q = locQ_G[threadIdx.x];
	tmp_nQ[Q + b * nQ] = c_nG[G + b * nG] * scale;
    }
}

template <typename Tcomplex, typename Treal>
__global__ void add_to_density(int nb,
			       int nR,
			       double* f_n,
			       Tcomplex* psit_nR,
			       double* rho_R)
{
    constexpr bool realtype = std::is_same<Tcomplex, Treal>::value;

    int R = threadIdx.x + blockIdx.x * blockDim.x;
    if (R < nR)
    {
	double rho = 0.0;
	for (int b=0; b< nb; b++)
	{
	    int idx = b * nR + R;
	    if constexpr(realtype) {
	    	rho += f_n[b] * double(psit_nR[idx] * psit_nR[idx]);
	    } else {
	    	rho += f_n[b] * double(psit_nR[idx].x * psit_nR[idx].x + psit_nR[idx].y * psit_nR[idx].y);
	    }
	}
	rho_R[R] += rho;
    }
}

template <typename Tcomplex, typename Treal>
__global__ void pw_insert(int nG,
			     int nQ,
			     Tcomplex* c_G,
			     npy_int32* Q_G,
			     Treal scale,
			     Tcomplex* tmp_Q)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    if (G < nG)
	tmp_Q[Q_G[G]] = c_G[G] * scale;
}

CLINKAGE void gpawDeviceSynchronize()
{
    gpuDeviceSynchronize();
}


CLINKAGE void add_to_density_gpu_launch_kernel(int nb,
					int nR,
					double* f_n,
					void* psit_nR,
					double* rho_R,
					int dtypenum,
					gpuStream_t stream)
{
    if (!nR) return;

	const dim3 blocks((nR+255)/256);
	const dim3 threads(256);
	const int shmem = 0;

    if (dtypenum==NP_DOUBLE_COMPLEX)
    {
		gpaw::launch_kernel(add_to_density<gpuDoubleComplex, double>, blocks, threads, shmem, stream,
			nb, nR, f_n, static_cast<gpuDoubleComplex*>(psit_nR), rho_R);
    }
    else if (dtypenum==NP_FLOAT_COMPLEX)
    {
		gpaw::launch_kernel(add_to_density<gpuFloatComplex, float>, blocks, threads, shmem, stream,
			nb, nR, f_n, static_cast<gpuFloatComplex*>(psit_nR), rho_R);
    }
	else if (dtypenum==NP_FLOAT)
    {
        gpaw::launch_kernel(add_to_density<float, float>, blocks, threads, shmem, stream,
			nb, nR, f_n, static_cast<float*>(psit_nR), rho_R);
    }
	else if (dtypenum==NP_DOUBLE)
    {
        gpaw::launch_kernel(add_to_density<double, double>, blocks, threads, shmem, stream,
			nb, nR, f_n, static_cast<double*>(psit_nR), rho_R);
    }
    else
    {
        assert(false);
    }
}

CLINKAGE void pw_amend_insert_realwf_gpu_launch_kernel(int dtypenum,
											  int nb,
                                              int nx,
                                              int ny,
                                              int nz,
                                              int n,
                                              int m,
                                              void* array_nQ,
											  gpuStream_t stream)
{
    if ((!nb) || (!max(n,m))) return;
	// FIXME which max() ?
	const dim3 blocks((nb+15)/16, (max(n,m)+15)/16);
	const dim3 threads(16, 16);
	const int shmem = 0;

	if (dtypenum == NP_DOUBLE_COMPLEX)
	{
		gpaw::launch_kernel(
			pw_amend_insert_realwf<gpuDoubleComplex>,
			blocks,
			threads,
			shmem,
			stream,
			nb,
			nx,
			ny,
			nz,
			n,
			m,
			static_cast<gpuDoubleComplex*>(array_nQ)
		);
	}
	else if (dtypenum == NP_FLOAT_COMPLEX)
	{
		gpaw::launch_kernel(
			pw_amend_insert_realwf<gpuFloatComplex>,
			blocks,
			threads,
			shmem,
			stream,
			nb,
			nx,
			ny,
			nz,
			n,
			m,
			static_cast<gpuFloatComplex*>(array_nQ)
		);
	}
	else
	{
		assert(false);
	}
}

CLINKAGE void pw_insert_gpu_launch_kernel(
			     int dtypenum,
			     int nb,
			     int nG,
			     int nQ,
			     void* c_nG,
			     npy_int32* Q_G,
			     double scale,
			     void* tmp_nQ,
                 int rx, int ry, int rz,
				 gpuStream_t stream)
{
    if ((!nG) || (!nb)) return;

	{
		// launch "pw_insert"
		const dim3 blocks((nG + 15) / 16, (nb + 15) / 16);
		const dim3 threads(16, 16);
		const int shmem = 0;
		if (nb == 1)
		{
			if (dtypenum == NP_DOUBLE_COMPLEX)
			{
				gpaw::launch_kernel(
					pw_insert<gpuDoubleComplex, double>,
					blocks,
					threads,
					shmem,
					stream,
					nG,
					nQ,
					static_cast<gpuDoubleComplex*>(c_nG),
					Q_G,
					scale,
					static_cast<gpuDoubleComplex*>(tmp_nQ)
				);
			}
			else if (dtypenum == NP_FLOAT_COMPLEX)
			{
				gpaw::launch_kernel(
					pw_insert<gpuFloatComplex, float>,
					blocks,
					threads,
					shmem,
					stream,
					nG,
					nQ,
					static_cast<gpuFloatComplex*>(c_nG),
					Q_G,
					static_cast<float>(scale),
					static_cast<gpuFloatComplex*>(tmp_nQ)
				);
			}
			else assert(false);
		}
		else
		{
			if (dtypenum == NP_DOUBLE_COMPLEX)
			{
				gpaw::launch_kernel(
					pw_insert_many<gpuDoubleComplex, double>,
					blocks,
					threads,
					shmem,
					stream,
					nb,
					nG,
					nQ,
					static_cast<gpuDoubleComplex*>(c_nG),
					Q_G,
					scale,
					static_cast<gpuDoubleComplex*>(tmp_nQ)
				);
			}
			else if (dtypenum == NP_FLOAT_COMPLEX)
			{
				gpaw::launch_kernel(
					pw_insert_many<gpuFloatComplex, float>,
					blocks,
					threads,
					shmem,
					stream,
					nb,
					nG,
					nQ,
					static_cast<gpuFloatComplex*>(c_nG),
					Q_G,
					static_cast<float>(scale),
					static_cast<gpuFloatComplex*>(tmp_nQ)
				);
			}
			else assert(false);
		}
	} // end "pw_insert"

    // We identify real wave functions by noting that number of cartesian planewaves
    // does not equal to real space grid size (because z_Q <- z_R // 2 + 1)
    if (rx * ry * rz != nQ)
    {
        int n = rx / 2 - 1;
        int m = ry / 2 - 1;

		// FIXME which max() ?
		const dim3 blocks((nb+15)/16, (max(n,m)+15)/16);
		const dim3 threads(16, 16);
		const int shmem = 0;

        // The rx, ry, rz are the sizes of the 3D version of Q array. Since
        // we are dealing with real wave functions, the convention is that
        // the last axis is actually z_R // 2 + 1.
		if (dtypenum == NP_DOUBLE_COMPLEX)
		{
			gpaw::launch_kernel(
				pw_amend_insert_realwf<gpuDoubleComplex>,
				blocks,
				threads,
				shmem,
				stream,
				nb, rx, ry, rz/2 + 1, n, m, static_cast<gpuDoubleComplex*>(tmp_nQ)
			);
		}
		else if (dtypenum == NP_FLOAT_COMPLEX)
		{
			gpaw::launch_kernel(
				pw_amend_insert_realwf<gpuFloatComplex>,
				blocks,
				threads,
				shmem,
				stream,
				nb, rx, ry, rz/2 + 1, n, m, static_cast<gpuFloatComplex*>(tmp_nQ)
			);
		}
    }
}

template <typename Tcomplex, typename Treal, bool strided, bool cc>
__global__ void pwlfc_expand_kernel(Treal* f_Gs,
				       Treal* Gk_Gv,
					   Treal* pos_av,
					   Tcomplex* eikR_a,
					   Treal *Y_GL,
				       int* l_s,
				       int* a_J,
				       int* s_J,
				       int* I_J,
				       Treal* f_GI,
				       int nG,
				       int nJ,
				       int nL,
				       int nI,
				       int natoms,
				       int nsplines)

{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    int J = threadIdx.y + blockIdx.y * blockDim.y;

	__shared__ Tcomplex imag_powers[4];
	if (threadIdx.y == 0 && threadIdx.x == 0)
		imag_powers[0] = {1.0,0};
	if (threadIdx.y == 0 && threadIdx.x == 1)
		imag_powers[1] = {0,-1.0};
	if (threadIdx.y == 0 && threadIdx.x == 2)
		imag_powers[2] = {-1.0,0};
	if (threadIdx.y == 0 && threadIdx.x == 3)
		imag_powers[3] = {0,1.0};
    __syncthreads();

	//Tcomplex imag_powers[4] = {{1.0,0},{0,-1.0},{-1.0,0},{0,1.0}};

    if ((G < nG) && (J < nJ))
    {
	f_Gs += G*nsplines;
	Gk_Gv += G*3;
	pos_av += a_J[J]*3;
	Treal GkPos = (Gk_Gv[0] * pos_av[0] +
		       	   Gk_Gv[1] * pos_av[1] +
		           Gk_Gv[2] * pos_av[2]);
	Tcomplex emiGR = {cos(GkPos), -sin(GkPos)};
	int s = s_J[J];
	int l = l_s[s];
	Y_GL += G*nL + l*l;
	Tcomplex f1 = emiGR * eikR_a[a_J[J]] * imag_powers[l % 4] * f_Gs[s];
	if constexpr(strided) {
		f_GI += G*nI*2 + I_J[J];
		for (int m = 0; m < 2 * l + 1; m++) {
	    	Tcomplex f = f1 * Y_GL[m];
	    	f_GI[0] = f.x;
			if constexpr(cc)
	    		f_GI[nI] = -f.y;
			else
				f_GI[nI] = f.y;
	    	f_GI++;
		}
	} else {
		f_GI += (G*nI + I_J[J])*2;
	    for (int m = 0; m < 2 * l + 1; m++) {
	        Tcomplex f = f1 * Y_GL[m];
	        *f_GI++ = f.x;
			if constexpr(cc)
				*f_GI++ = -f.y;
			else
				*f_GI++ = f.y;
	        //*f_GI++ = cc ? -f.y : f.y;
   	    }
	}
    }
}

template <typename Tcomplex, typename Treal>
__global__ void dH_aii_times_P_ani(int nA, int nn, int nI,
				      npy_int32* ni_a,
					  Treal* dH_aii_dev,
				      Tcomplex* P_ani_dev,
				      Tcomplex* outP_ani_dev)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (n1 < nn) {
	Treal* dH_ii = dH_aii_dev;
	int I = 0;
	for (int a=0; a< nA; a++)
	{
	    int ni = ni_a[a];
	    int Istart = I;
	    for (int i=0; i< ni; i++)
	    {
		Tcomplex* outP_ni = outP_ani_dev + n1 * nI + I;
		Tcomplex result;
		if  constexpr (std::is_same<Tcomplex, Treal>::value) {
			result = 0.0;
		} else {
			result = {0.0, 0.0};
		}
		Tcomplex* P_ni = P_ani_dev + n1 * nI + Istart;
		for (int i2=0; i2 < ni; i2++)
		{
		   result = result + *P_ni * dH_ii[i2 * ni + i];
		   P_ni++;
		}
		*outP_ni = result;
		//outP_ni->x = result.x;
		//outP_ni->y = result.y;
		I++;
	    }
	    dH_ii += ni * ni;
	}
    }
}

template <unsigned int blockSize, typename Treal>
__device__ void warpReduce(volatile Treal *sdata, unsigned int tid) {
if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


// One block will always sum one G-vector. Thus, no block wide reduce.
template <unsigned int blockSize, typename Treal>
__global__ void pw_norm_kinetic_kernel(int nx, int nG,
                                       Treal* result_x,
                                       Treal* C_xG,
                                       Treal* kin_G)
{
	// Double check this line (and next)
	extern __shared__ __align__(sizeof(double)) unsigned char my_sdata[];
	Treal *sdata = reinterpret_cast<Treal *>(my_sdata);
    unsigned int tid = threadIdx.x;

    sdata[tid] = 0;
    unsigned int x = blockIdx.x;

    Treal* C_G = C_xG + (x * nG * 2); // C_xG is a Treal complex array
    unsigned int i = tid;
    while (i < nG)
    {
        Treal kin_i = kin_G[i] * (C_G[i*2] * C_G[i*2] + C_G[i*2+1] * C_G[i*2+1]);
        sdata[tid] += kin_i;
        i += blockSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize, Treal>(sdata, tid);
    if (tid == 0) result_x[x] = sdata[0];
}

template <unsigned int blockSize, typename Treal>
__global__ void pw_norm_kernel(int nx, int nG,
                               Treal* result_x,
                               Treal* C_xG)
{
    extern __shared__ __align__(sizeof(double)) unsigned char my_sdata[];
	Treal *sdata = reinterpret_cast<Treal *>(my_sdata);
    unsigned int tid = threadIdx.x;

    sdata[tid] = 0;
    unsigned int x = blockIdx.x;

    Treal* C_G = C_xG + (x * nG * 2); // C_xG is a double complex array
    unsigned int i = tid;
    while (i < nG)
    {
        Treal kin_i = C_G[i*2] * C_G[i*2] + C_G[i*2+1] * C_G[i*2+1];
        sdata[tid] += kin_i;
        i += blockSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize, Treal>(sdata, tid);
    if (tid == 0) result_x[x] = sdata[0];
}

CLINKAGE void dH_aii_times_P_ani_launch_kernel(int dtypenum,
					int nA, int nn,
					int nI, npy_int32* ni_a,
					void* dH_aii_dev,
					void* P_ani_dev,
					void* outP_ani_dev,
					gpuStream_t stream)
{
    if (!nn) return;

	const dim3 blocks((nn+255)/256);
	const dim3 threads(256);
	const int shmem = 0;

    if (dtypenum == NP_DOUBLE_COMPLEX)
    {
		gpaw::launch_kernel(
			dH_aii_times_P_ani<gpuDoubleComplex, double>,
			blocks,
			threads,
			shmem,
			stream,
			nA, nn, nI, ni_a,
			static_cast<double*>(dH_aii_dev),
			static_cast<gpuDoubleComplex*>(P_ani_dev),
			static_cast<gpuDoubleComplex*>(outP_ani_dev)
		);
    }
    else if (dtypenum == NP_FLOAT_COMPLEX)
	{
		gpaw::launch_kernel(
			dH_aii_times_P_ani<gpuFloatComplex, float>,
			blocks,
			threads,
			shmem,
			stream,
			nA, nn, nI, ni_a,
			static_cast<float*>(dH_aii_dev),
			static_cast<gpuFloatComplex*>(P_ani_dev),
			static_cast<gpuFloatComplex*>(outP_ani_dev)
		);
	}
	else if (dtypenum == NP_DOUBLE)
    {
		gpaw::launch_kernel(
			dH_aii_times_P_ani<double, double>,
			blocks,
			threads,
			shmem,
			stream,
			nA, nn, nI, ni_a,
			static_cast<double*>(dH_aii_dev),
			static_cast<double*>(P_ani_dev),
			static_cast<double*>(outP_ani_dev)
		);
    }
	else if (dtypenum == NP_FLOAT)
	{
		gpaw::launch_kernel(
			dH_aii_times_P_ani<float, float>,
			blocks,
			threads,
			shmem,
			stream,
			nA, nn, nI, ni_a,
			static_cast<float*>(dH_aii_dev),
			static_cast<float*>(P_ani_dev),
			static_cast<float*>(outP_ani_dev)
		);
	}
	else assert(false);
}

CLINKAGE void pw_norm_gpu_launch_kernel(int dtypenum,
										int nx, int nG,
										void* result_x,
										void* C_xG,
										gpuStream_t stream)
{
	if (!nx) return;

	const dim3 blocks(nx, 1);
	const dim3 threads(512, 1);
	const int shmem = sizeof(double) * 512;

	if (dtypenum == NP_DOUBLE_COMPLEX)
	{
		gpaw::launch_kernel(
			pw_norm_kernel<512, double>,
			blocks,
			threads,
			shmem,
			stream,
			nx,
			nG,
			static_cast<double*>(result_x),
			static_cast<double*>(C_xG)
		);
	}
	else if (dtypenum == NP_FLOAT_COMPLEX)
	{
		gpaw::launch_kernel(
			pw_norm_kernel<512, float>,
			blocks,
			threads,
			shmem,
			stream,
			nx,
			nG,
			static_cast<float*>(result_x),
			static_cast<float*>(C_xG)
		);
	} else assert(false);
}

CLINKAGE void pw_norm_kinetic_gpu_launch_kernel(int dtypenum,
												int nx, int nG,
												void* result_x,
												void* C_xG,
												void* kin_G,
												gpuStream_t stream)
{
	if (!nx) return;

	const dim3 blocks(nx, 1);
	const dim3 threads(512, 1);
	const int shmem = sizeof(double) * 512;

	if (dtypenum == NP_DOUBLE_COMPLEX)
	{
		gpaw::launch_kernel(
			pw_norm_kinetic_kernel<512, double>,
			blocks,
			threads,
			shmem,
			stream,
			nx,
			nG,
			static_cast<double*>(result_x),
			static_cast<double*>(C_xG),
			static_cast<double*>(kin_G)
		);
	}
	else if (dtypenum == NP_FLOAT_COMPLEX)
	{
		gpaw::launch_kernel(
			pw_norm_kinetic_kernel<512, float>,
			blocks,
			threads,
			shmem,
			stream,
			nx,
			nG,
			static_cast<float*>(result_x),
			static_cast<float*>(C_xG),
			static_cast<float*>(kin_G)
		);
	}
	else assert(false);
}


CLINKAGE void pwlfc_expand_gpu_launch_kernel(int dtypenum,
				    void* f_Gs,
					void* Gk_Gv,
					void* pos_av,
					void* eikR_a,
				    void *Y_GL,
				    int* l_s,
				    int* a_J,
				    int* s_J,
				    void* f_GI,
				    int* I_J,
				    int nG,
				    int nJ,
				    int nL,
				    int nI,
				    int natoms,
				    int nsplines,
				    bool cc,
					gpuStream_t stream)
{
    if ((!nG) || (!nJ)) return;

	const dim3 blocks((nG+15)/16, (nJ+15)/16); // blockDimX must be > 4 due to shared initialization,
	const dim3 threads(16, 16);
	const int shmem = 0;

	// FIXME way too much copy-pasting going on here. Kernel args for NP_DOUBLE_COMPLEX and NP_DOUBLE are identical!

	if (dtypenum == NP_DOUBLE_COMPLEX)
	{
		auto kernel = pwlfc_expand_kernel<gpuDoubleComplex, double, false, false>;
		if (cc)
		{
			kernel = pwlfc_expand_kernel<gpuDoubleComplex, double, false, true>;
		}

		gpaw::launch_kernel(
			kernel,
			blocks,
			threads,
			shmem,
			stream,
			static_cast<double*>(f_Gs),
			static_cast<double*>(Gk_Gv),
			static_cast<double*>(pos_av),
			static_cast<gpuDoubleComplex*>(eikR_a),
			static_cast<double*>(Y_GL),
			l_s,
			a_J,
			s_J,
			I_J,
			static_cast<double*>(f_GI),
			nG,
			nJ,
			nL,
			nI,
			natoms,
			nsplines
		);
	}
	else if (dtypenum == NP_DOUBLE)
	{
		auto kernel = pwlfc_expand_kernel<gpuDoubleComplex, double, true, false>;
		if (cc)
		{
			kernel = pwlfc_expand_kernel<gpuDoubleComplex, double, true, true>;
		}

		gpaw::launch_kernel(
			kernel,
			blocks,
			threads,
			shmem,
			stream,
			static_cast<double*>(f_Gs),
			static_cast<double*>(Gk_Gv),
			static_cast<double*>(pos_av),
			static_cast<gpuDoubleComplex*>(eikR_a),
			static_cast<double*>(Y_GL),
			l_s,
			a_J,
			s_J,
			I_J,
			static_cast<double*>(f_GI),
			nG,
			nJ,
			nL,
			nI,
			natoms,
			nsplines
		);
	}
	else if (dtypenum == NP_FLOAT_COMPLEX)
	{
		auto kernel = pwlfc_expand_kernel<gpuFloatComplex, float, false, false>;
		if (cc)
		{
			kernel = pwlfc_expand_kernel<gpuFloatComplex, float, false, true>;
		}

		gpaw::launch_kernel(
			kernel,
			blocks,
			threads,
			shmem,
			stream,
			static_cast<float*>(f_Gs),
			static_cast<float*>(Gk_Gv),
			static_cast<float*>(pos_av),
			static_cast<gpuFloatComplex*>(eikR_a),
			static_cast<float*>(Y_GL),
			l_s,
			a_J,
			s_J,
			I_J,
			static_cast<float*>(f_GI),
			nG,
			nJ,
			nL,
			nI,
			natoms,
			nsplines
		);
	}
	else if (dtypenum == NP_FLOAT)
	{
		auto kernel = pwlfc_expand_kernel<gpuFloatComplex, float, true, false>;
		if (cc)
		{
			kernel = pwlfc_expand_kernel<gpuFloatComplex, float, true, true>;
		}

		gpaw::launch_kernel(
			kernel,
			blocks,
			threads,
			shmem,
			stream,
			static_cast<float*>(f_Gs),
			static_cast<float*>(Gk_Gv),
			static_cast<float*>(pos_av),
			static_cast<gpuFloatComplex*>(eikR_a),
			static_cast<float*>(Y_GL),
			l_s,
			a_J,
			s_J,
			I_J,
			static_cast<float*>(f_GI),
			nG,
			nJ,
			nL,
			nI,
			natoms,
			nsplines
		);
	}
	//gpuDeviceSynchronize();
}
