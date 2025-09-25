from typing import TYPE_CHECKING

import gpaw.cgpaw as cgpaw
from gpaw import GPAW_NO_C_EXTENSION
from gpaw.new.timer import trace

__all__ = ['GPU_AWARE_MPI']

GPU_AWARE_MPI = getattr(cgpaw, 'gpu_aware_mpi', False)
GPU_ENABLED = getattr(cgpaw, 'GPU_ENABLED', False)

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

        from cupy.cuda.stream import get_current_stream
        import functools

        def s(fun):
            @functools.wraps(fun)
            def wrapper(*args):
                return fun(*args, get_current_stream().ptr)
            return wrapper

        add_to_density_gpu = s(add_to_density_gpu)
        calculate_residuals_gpu = s(calculate_residuals_gpu)
        dH_aii_times_P_ani_gpu = s(dH_aii_times_P_ani_gpu)
        evaluate_lda_gpu = s(evaluate_lda_gpu)
        evaluate_pbe_gpu = s(evaluate_pbe_gpu)
        pw_amend_insert_realwf_gpu = s(pw_amend_insert_realwf_gpu)
        pw_insert_gpu = s(pw_insert_gpu)
        pwlfc_expand_gpu = s(pwlfc_expand_gpu)
        pw_norm_kinetic_gpu = s(pw_norm_kinetic_gpu)
        pw_norm_gpu = s(pw_norm_gpu)

        w = trace(gpu=True)
        add_to_density_gpu = w(add_to_density_gpu)
        calculate_residuals_gpu = w(calculate_residuals_gpu)
        dH_aii_times_P_ani_gpu = w(dH_aii_times_P_ani_gpu)
        evaluate_lda_gpu = w(evaluate_lda_gpu)
        evaluate_pbe_gpu = w(evaluate_pbe_gpu)
        pw_amend_insert_realwf_gpu = w(pw_amend_insert_realwf_gpu)
        pw_insert_gpu = w(pw_insert_gpu)
        pwlfc_expand_gpu = w(pwlfc_expand_gpu)
        pw_norm_kinetic_gpu = w(pw_norm_kinetic_gpu)
        pw_norm_gpu = w(pw_norm_gpu)
    else:
        from gpaw.purepython import (add_to_density_gpu,
                                     calculate_residuals_gpu,  # noqa
                                     dH_aii_times_P_ani_gpu, evaluate_lda_gpu,
                                     evaluate_pbe_gpu,
                                     pw_amend_insert_realwf_gpu,
                                     pw_insert_gpu, pwlfc_expand_gpu,
                                     pw_norm_kinetic_gpu, pw_norm_gpu)
else:
    from gpaw.purepython import (add_to_density, pw_insert, pw_precond,  # noqa
                                 pwlfc_expand, symmetrize_ft)
    from gpaw.purepython import (add_to_density_gpu,  # noqa
                                 calculate_residuals_gpu,
                                 dH_aii_times_P_ani_gpu, evaluate_lda_gpu,
                                 evaluate_pbe_gpu,
                                 pw_amend_insert_realwf_gpu,
                                 pw_insert_gpu, pwlfc_expand_gpu,
                                 pw_norm_kinetic_gpu, pw_norm_gpu)
