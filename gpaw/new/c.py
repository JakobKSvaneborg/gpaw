from typing import TYPE_CHECKING

import numpy as np

import gpaw.cgpaw as cgpaw
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.typing import Array1D, ArrayND
from gpaw.utilities import as_complex_dtype, as_real_dtype
from gpaw import GPAW_NO_C_EXTENSION

__all__ = ['GPU_AWARE_MPI']

GPU_AWARE_MPI = getattr(cgpaw, 'gpu_aware_mpi', False)
GPU_ENABLED = getattr(cgpaw, 'GPU_ENABLED', False)


def add_to_density(f: float,
                   psit_X: ArrayND,
                   nt_X: ArrayND) -> None:
    nt_X += f * abs(psit_X)**2


def pw_precond(G2_G: Array1D,
               r_G: Array1D,
               ekin: float,
               o_G: Array1D) -> None:
    x = 1 / ekin / 3 * G2_G
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    o_G[:] = -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * r_G


def pw_insert(coef_G: Array1D,
              Q_G: Array1D,
              x: float,
              array_Q: Array1D) -> None:
    array_Q[:] = 0.0
    array_Q.ravel()[Q_G] = x * coef_G


def pw_insert_gpu(psit_nG,
                  Q_G,
                  scale,
                  psit_bQ,
                  nx, ny, nz):
    from _gpaw import pw_insert_gpu as evalf
    evalf(psit_nG, Q_G, scale, psit_bQ, nx, ny, nz)
    
    #assert scale == 1.0
    #psit_bQ[..., Q_G] = psit_nG
    #if nx * ny * nz != psit_bQ.shape[-1]:
    #    n, m = nx // 2 - 1, ny // 2 - 1
    #    pw_amend_insert_realwf_gpu(psit_bQ.reshape((-1, nx, ny, nz // 2 + 1)),
    #                               n, m)


def pwlfc_expand(f_Gs, emiGR_Ga, Y_GL,
                 l_s, a_J, s_J,
                 cc, f_GI):
    real = np.issubdtype(f_GI.dtype, np.floating)
    I1 = 0
    for J, (a, s) in enumerate(zip(a_J, s_J)):
        l = l_s[s]
        I2 = I1 + 2 * l + 1
        f_Gi = (f_Gs[:, s] *
                emiGR_Ga[:, a] *
                Y_GL[:, l**2:(l + 1)**2].T *
                (-1.0j)**l).T
        if cc:
            np.conjugate(f_Gi, f_Gi)
        if real:
            f_GI[::2, I1:I2] = f_Gi.real
            f_GI[1::2, I1:I2] = f_Gi.imag
        else:
            f_GI[:, I1:I2] = f_Gi
        I1 = I2


def pwlfc_expand_gpu(f_Gs, emiGR_Ga, Y_GL,
                     l_s, a_J, s_J,
                     cc, f_GI, I_J):
    from _gpaw import pwlfc_expand_gpu as expand
    expand(f_Gs, emiGR_Ga, Y_GL,
        l_s, a_J, s_J,
        cc, f_GI, I_J)

    #pwlfc_expand(f_Gs, emiGR_Ga, Y_GL,
    #             l_s, a_J, s_J,
    #             cc, f_GI)


def dH_aii_times_P_ani_gpu(dH_aii, ni_a,
                           P_nI, out_nI):
    from _gpaw import dH_aii_times_P_ani_gpu as evalf
    if not dH_aii.dtype == as_real_dtype(P_nI.dtype):
        breakpoint()
    evalf(dH_aii, ni_a, P_nI, out_nI)
    
    #I1 = 0
    #J1 = 0
    #for ni in ni_a.get():
    #    I2 = I1 + ni
    #    J2 = J1 + ni**2
    #    dH_ii = dH_aii[J1:J2].reshape((ni, ni))
    #    out_nI[:, I1:I2] = P_nI[:, I1:I2] @ dH_ii
    #    I1 = I2
    #    J1 = J2


def pw_amend_insert_realwf_gpu(array_nQ, n, m):
    from _gpaw import pw_amend_insert_realwf_gpu as evalf
    evalf(array_nQ, n, m)
    
    #for array_Q in array_nQ:
    #    t = array_Q[:, :, 0]
    #    t[0, -m:] = t[0, m:0:-1].conj()
    #    t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
    #    t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
    #    t[-n:, 0] = t[n:0:-1, 0].conj()


def calculate_residuals_gpu(residual_nG, eps_n, wfs_nG):
    from _gpaw import calculate_residuals_gpu as evalf
    evalf(residual_nG, eps_n, wfs_nG)
    
    #for residual_G, eps, wfs_G in zip(residual_nG, eps_n, wfs_nG):
    #    residual_G -= eps * wfs_G


def add_to_density_gpu(weight_n, psit_nR, nt_R):
    #for weight, psit_R in zip(weight_n, psit_nR):
    #    nt_R += float(weight) * cp.abs(psit_R)**2
    from _gpaw import add_to_density_gpu as evalf
    evalf(weight_n, psit_nR, nt_R)


def symmetrize_ft(a_R, b_R, r_cc, t_c, offset_c):
    if (r_cc == np.eye(3, dtype=int)).all() and not t_c.any():
        b_R[:] = a_R
        return
    raise NotImplementedError


def evaluate_lda_gpu(nt_sr, vxct_sr, e_r) -> None:
    if cupy_is_fake:
        from gpaw.xc.kernel import XCKernel
        XCKernel('LDA').calculate(e_r._data, nt_sr._data, vxct_sr._data)
    else:
        from _gpaw import evaluate_lda_gpu as evalf  # type: ignore
        evalf(nt_sr, vxct_sr, e_r)


def evaluate_pbe_gpu(nt_sr, vxct_sr, e_r, sigma_xr, dedsigma_xr) -> None:
    from gpaw.xc.kernel import XCKernel
    XCKernel('PBE').calculate(e_r._data, nt_sr._data, vxct_sr._data,
                              sigma_xr._data, dedsigma_xr._data)


def pw_norm_gpu(result_x, C_xG):
    if cupy_is_fake:
        result_x._data[:] = np.sum(np.abs(C_xG._data)**2, axis=1)
    else:
        from _gpaw import pw_norm_gpu as evalf
        evalf(result_x, C_xG)
        #result_x[:] = cp.sum(cp.abs(C_xG)**2, axis=1)


def pw_norm_kinetic_gpu(result_x, a_xG, kin_G):
    if cupy_is_fake:
        result_x._data[:] = np.sum(
            np.abs(a_xG._data)**2 * kin_G._data[None, :],
            axis=1)
    else:
        from _gpaw import pw_norm_kinetic_gpu as evalf
        evalf(result_x, a_xG, kin_G)
        #result_x[:] = cp.sum(cp.abs(a_xG)**2 * kin_G[None, :], axis=1)


if not TYPE_CHECKING and not GPAW_NO_C_EXTENSION:
    from gpaw.cgpaw import (add_to_density, pw_insert, pw_precond,  # noqa
                            pwlfc_expand, symmetrize_ft)

    if GPU_ENABLED:
        from gpaw.cgpaw import add_to_density_gpu  # noqa
        from gpaw.cgpaw import (calculate_residuals_gpu,  # noqa
                                dH_aii_times_P_ani_gpu, evaluate_lda_gpu,
                                evaluate_pbe_gpu, pw_amend_insert_realwf_gpu,
                                pw_insert_gpu, pwlfc_expand_gpu,
                                pw_norm_kinetic_gpu, pw_norm_gpu)
