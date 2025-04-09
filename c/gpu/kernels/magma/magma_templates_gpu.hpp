#pragma once

#include "magma_gpaw.hpp"
#include "template_utils.hpp"
#include "../../gpu-runtime.h"

template<typename T>
magma_int_t magma_Xsyevd_gpu(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, T* matrix,
    magma_int_t lda, T* eigvals, T* wA, magma_int_t ldwa, T* work, magma_int_t lwork, magma_int_t* iwork,
    magma_int_t liwork, magma_int_t* info)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return magma_ssyevd_gpu(jobz, uplo, n, matrix, lda, eigvals, wA, ldwa, work, lwork, iwork, liwork, info);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return magma_dsyevd_gpu(jobz, uplo, n, matrix, lda, eigvals, wA, ldwa, work, lwork, iwork, liwork, info);
    }
    else gpaw::static_no_match();
}

template<typename RealT>
magma_int_t magma_Xheevd_gpu(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, magmaComplex<RealT>* matrix,
    magma_int_t lda, RealT* eigvals, magmaComplex<RealT>* wA, magma_int_t ldwa, magmaComplex<RealT>* work,
    magma_int_t lwork, RealT* rwork, magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info)
{
    if constexpr (std::is_same_v<RealT, float>)
    {
        return magma_cheevd_gpu(jobz, uplo, n, matrix, lda, eigvals, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info);
    }
    else if constexpr (std::is_same_v<RealT, double>)
    {
        return magma_zheevd_gpu(jobz, uplo, n, matrix, lda, eigvals, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info);
    }
    else gpaw::static_no_match();
}

template<typename T>
magma_int_t magma_Xsyevd_gpu(const MagmaEighContext& context, T* matrix, T* eigvals, SyevdWorkspace_gpu<T> &workspace, bool do_workspace_query)
{
    magma_int_t info;

    if (do_workspace_query)
    {
        T work_temp;
        magma_int_t iwork_temp;

        // ldwa does not get queried, so we just fix it to be the same as matrix lda
        if (workspace.ldwa <= 0)
        {
            workspace.ldwa = context.matrix_lda;
        }

        magma_Xsyevd_gpu<T>(context.jobz, context.uplo, context.matrix_size, nullptr, context.matrix_lda,
            nullptr, nullptr, workspace.ldwa, &work_temp, -1, &iwork_temp, -1, &info
        );

        workspace.lwork = static_cast<magma_int_t>(work_temp);
        workspace.liwork = static_cast<magma_int_t>(iwork_temp);

        return info;
    }

    return magma_Xsyevd_gpu<T>(context.jobz, context.uplo, context.matrix_size, matrix, context.matrix_lda,
        eigvals, workspace.wA, workspace.ldwa, workspace.work, workspace.lwork, workspace.iwork, workspace.liwork, &info
    );
}

template<typename RealT>
magma_int_t magma_Xheevd_gpu(const MagmaEighContext& context, magmaComplex<RealT>* matrix, RealT* eigvals,
    HeevdWorkspace_gpu<RealT> &workspace, bool do_workspace_query)
{
    magma_int_t info;

    if (do_workspace_query)
    {
        magmaComplex<RealT> work_temp;
        RealT rwork_temp;
        magma_int_t iwork_temp;

        if (workspace.ldwa <= 0)
        {
            workspace.ldwa = context.matrix_lda;
        }

        magma_Xheevd_gpu<RealT>(context.jobz, context.uplo, context.matrix_size, nullptr, context.matrix_lda,
            nullptr, nullptr, workspace.ldwa, &work_temp, -1, &rwork_temp, -1, &iwork_temp, -1, &info
        );

        workspace.lwork = static_cast<magma_int_t>(MAGMA_Z_REAL(work_temp));
        workspace.lrwork = static_cast<magma_int_t>(rwork_temp);
        workspace.liwork = iwork_temp;

        return info;
    }

    return magma_Xheevd_gpu<RealT>(context.jobz, context.uplo, context.matrix_size, matrix, context.matrix_lda,
        eigvals, workspace.wA, workspace.ldwa, workspace.work, workspace.lwork, workspace.rwork, workspace.lrwork,
        workspace.iwork, workspace.liwork, &info
    );
}


template<typename T>
EighErrorType magma_symmetric_solver_gpu(
    const MagmaEighContext& context,
    const T* const in_matrix,
    T* inout_eigvals,
    T* inout_eigvecs)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");

    const size_t n = context.matrix_size;

    // Magma overrides the input arrays with eigenvectors, so we operate on a copy
    gpuMemcpy(inout_eigvecs, in_matrix, n*n*sizeof(T), gpuMemcpyDeviceToDevice);

    // Query
    SyevdWorkspace_gpu<T> workspace;
    MAGMA_CHECK(magma_Xsyevd_gpu<T>(context, nullptr, nullptr, workspace, true));

    // All work buffers are on HOST. Magma also needs the eigenvalue buffer on host
    MAGMA_CHECK(magma_host_malloc(&workspace.work, static_cast<size_t>(workspace.lwork)));
    MAGMA_CHECK(magma_host_malloc(&workspace.iwork, static_cast<size_t>(workspace.liwork)));
    MAGMA_CHECK(magma_host_malloc(&workspace.wA, n * static_cast<size_t>(workspace.ldwa)));

    T* h_eigvals;
    MAGMA_CHECK(magma_host_malloc(&h_eigvals, n));

    const magma_int_t status = magma_Xsyevd_gpu<T>(context, inout_eigvecs, inout_eigvals, workspace, false);

    // Copy eigenvalues back to device
    gpuMemcpy(inout_eigvals, h_eigvals, n*sizeof(T), gpuMemcpyHostToDevice);

    MAGMA_CHECK(magma_host_free(h_eigvals));
    MAGMA_CHECK(magma_host_free(workspace.wA));
    MAGMA_CHECK(magma_host_free(workspace.iwork));
    MAGMA_CHECK(magma_host_free(workspace.work));

    return interpret_magma_status(status);
}

template<typename T>
EighErrorType magma_hermitian_solver_gpu(
    const MagmaEighContext& context,
    const magmaComplex<T>* const in_matrix,
    T* inout_eigvals,
    magmaComplex<T>* inout_eigvecs)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");

    const size_t n = context.matrix_size;

    // Magma overrides the input arrays with eigenvectors, so we operate on a copy
    gpuMemcpy(inout_eigvecs, in_matrix, n*n*sizeof(magmaComplex<T>), gpuMemcpyDeviceToDevice);

    // Query
    HeevdWorkspace_gpu<T> workspace;
    MAGMA_CHECK(magma_Xheevd_gpu<T>(context, nullptr, nullptr, workspace, true));

    // All work buffers are on HOST. Magma also needs the eigenvalue buffer on host
    MAGMA_CHECK(magma_host_malloc(&workspace.work, static_cast<size_t>(workspace.lwork)));
    MAGMA_CHECK(magma_host_malloc(&workspace.iwork, static_cast<size_t>(workspace.liwork)));
    MAGMA_CHECK(magma_host_malloc(&workspace.wA, n * static_cast<size_t>(workspace.ldwa)));
    MAGMA_CHECK(magma_host_malloc(&workspace.rwork, static_cast<size_t>(workspace.lrwork)));

    T* h_eigvals;
    MAGMA_CHECK(magma_host_malloc(&h_eigvals, n));

    const magma_int_t status = magma_Xheevd_gpu<T>(context, inout_eigvecs, inout_eigvals, workspace, false);

    // Copy eigenvalues back to device
    gpuMemcpy(inout_eigvals, h_eigvals, n*sizeof(T), gpuMemcpyHostToDevice);

    MAGMA_CHECK(magma_host_free(h_eigvals));
    MAGMA_CHECK(magma_host_free(workspace.rwork));
    MAGMA_CHECK(magma_host_free(workspace.wA));
    MAGMA_CHECK(magma_host_free(workspace.iwork));
    MAGMA_CHECK(magma_host_free(workspace.work));

    return interpret_magma_status(status);
}
