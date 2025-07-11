#pragma once

#include "magma_gpaw.hpp"
#include "../template_utils.hpp"
#include "../../gpu-runtime.h"

// Templated in-place transpose, works for both real and complex matrices
template<typename T>
void magmablas_Xtranspose_inplace(magma_int_t matrix_size, T* matrix, magma_int_t lda, magma_queue_t queue)
{
    if constexpr (std::is_same_v<T, float>)
    {
        magmablas_stranspose_inplace(matrix_size, matrix, lda, queue);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        magmablas_dtranspose_inplace(matrix_size, matrix, lda, queue);
    }
    else if constexpr (std::is_same_v<T, magmaComplex<float>>)
    {
        magmablas_ctranspose_inplace(matrix_size, matrix, lda, queue);
    }
    else if constexpr (std::is_same_v<T, magmaComplex<double>>)
    {
        magmablas_ztranspose_inplace(matrix_size, matrix, lda, queue);
    }
    else gpaw::static_no_match();
}

// Templated in-place Hermitian conjugation
template<typename RealT>
void magmablas_Xtranspose_conj_inplace(magma_int_t matrix_size, magmaComplex<RealT>* matrix, magma_int_t lda, magma_queue_t queue)
{
    if constexpr (std::is_same_v<RealT, float>)
    {
        magmablas_ctranspose_conj_inplace(matrix_size, matrix, lda, queue);
    }
    else if constexpr (std::is_same_v<RealT, double>)
    {
        magmablas_ztranspose_conj_inplace(matrix_size, matrix, lda, queue);
    }
    else gpaw::static_no_match();
}

// Templated magma_ssyevd or magma_dsyevd
template<typename T>
magma_int_t magma_Xsyevd(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, T* matrix,
    magma_int_t lda, T* eigvals, T* work, magma_int_t lwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return magma_ssyevd(jobz, uplo, n, matrix, lda, eigvals, work, lwork, iwork, liwork, info);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return magma_dsyevd(jobz, uplo, n, matrix, lda, eigvals, work, lwork, iwork, liwork, info);
    }
    else gpaw::static_no_match();
}

// Templated magma_cheevd or magma_zheevd
template<typename RealT>
magma_int_t magma_Xheevd(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, magmaComplex<RealT>* matrix,
    magma_int_t lda, RealT* eigvals, magmaComplex<RealT>* work, magma_int_t lwork, RealT* rwork, magma_int_t lrwork,
    magma_int_t* iwork, magma_int_t liwork, magma_int_t* info)
{
    if constexpr (std::is_same_v<RealT, float>)
    {
        return magma_cheevd(jobz, uplo, n, matrix, lda, eigvals, work, lwork, rwork, lrwork, iwork, liwork, info);
    }
    else if constexpr (std::is_same_v<RealT, double>)
    {
        return magma_zheevd(jobz, uplo, n, matrix, lda, eigvals, work, lwork, rwork, lrwork, iwork, liwork, info);
    }
    else gpaw::static_no_match();
}

/* Templated magma syevd. Overrides the input matrix.
If do_workspace_query is set, does instead a workspace query and fills in the optimal work sizes in the input workspace struct.
*/
template<typename T>
magma_int_t magma_Xsyevd(const MagmaEighContext& context, T* matrix, T* eigvals, SyevdWorkspace<T> &workspace, bool do_workspace_query)
{
    magma_int_t info;

    if (do_workspace_query)
    {
        T work_temp;
        magma_int_t iwork_temp;

        magma_Xsyevd<T>(context.jobz, context.uplo, context.matrix_size, nullptr, context.matrix_lda,
            nullptr, &work_temp, -1, &iwork_temp, -1, &info
        );

        workspace.lwork = static_cast<magma_int_t>(work_temp);
        workspace.liwork = static_cast<magma_int_t>(iwork_temp);

        return info;
    }

    return magma_Xsyevd<T>(context.jobz, context.uplo, context.matrix_size, matrix, context.matrix_lda,
        eigvals, workspace.work, workspace.lwork, workspace.iwork, workspace.liwork, &info
    );
}

/* Templated magma heevd. Overrides the input matrix.
If do_workspace_query is set, does instead a workspace query and fills in the optimal work sizes in the input workspace struct.
*/
template<typename RealT>
magma_int_t magma_Xheevd(const MagmaEighContext& context, magmaComplex<RealT>* matrix, RealT* eigvals,
    HeevdWorkspace<RealT> &workspace, bool do_workspace_query)
{
    magma_int_t info;

    if (do_workspace_query)
    {
        magmaComplex<RealT> work_temp;
        RealT rwork_temp;
        magma_int_t iwork_temp;

        magma_Xheevd<RealT>(context.jobz, context.uplo, context.matrix_size, nullptr, context.matrix_lda,
            nullptr, &work_temp, -1, &rwork_temp, -1, &iwork_temp, -1, &info
        );

        workspace.lwork = static_cast<magma_int_t>(MAGMA_Z_REAL(work_temp));
        workspace.lrwork = static_cast<magma_int_t>(rwork_temp);
        workspace.liwork = iwork_temp;

        return info;
    }

    return magma_Xheevd<RealT>(context.jobz, context.uplo, context.matrix_size, matrix, context.matrix_lda,
        eigvals, workspace.work, workspace.lwork, workspace.rwork, workspace.lrwork, workspace.iwork, workspace.liwork, &info
    );
}

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
