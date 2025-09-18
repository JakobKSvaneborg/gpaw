#pragma once

#include "magma_gpaw.hpp"
#include "../utils.hpp"

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
EighErrorType magma_symmetric_solver_cpu(
    const MagmaEighContext& context,
    const T* const in_matrix,
    T* inout_eigvals,
    T* inout_eigvecs)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");

    const size_t n = context.matrix_size;

    // Magma overrides the input arrays with eigenvectors, so we operate on a copy
    memcpy(inout_eigvecs, in_matrix, n*n*sizeof(T));

    SyevdWorkspace<T> workspace;
    bool bQuery = true;
    MAGMA_CHECK(magma_Xsyevd<T>(context, nullptr, nullptr, workspace, bQuery));

    workspace.work = gpaw::TMalloc<T>(static_cast<size_t>(workspace.lwork));
    workspace.iwork = gpaw::TMalloc<magma_int_t>(static_cast<size_t>(workspace.liwork));

    bQuery = false;
    const magma_int_t status = magma_Xsyevd<T>(context, inout_eigvecs, inout_eigvals, workspace, bQuery);

    gpaw::TFree(workspace.work);
    gpaw::TFree(workspace.iwork);

    return interpret_magma_status(status);
}

template<typename T>
EighErrorType magma_hermitian_solver_cpu(
    const MagmaEighContext& context,
    const magmaComplex<T>* const in_matrix,
    T* inout_eigvals,
    magmaComplex<T>* inout_eigvecs)
{
    // Note that T here is a real type
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");

    const size_t n = context.matrix_size;
    memcpy(inout_eigvecs, in_matrix, n*n*sizeof(magmaComplex<T>));

    // Workspace query
    HeevdWorkspace<T> workspace;
    MAGMA_CHECK(magma_Xheevd<T>(context, nullptr, nullptr, workspace, true));

    workspace.work = gpaw::TMalloc<magmaComplex<T>>(static_cast<size_t>(workspace.lwork));
    workspace.iwork = gpaw::TMalloc<magma_int_t>(static_cast<size_t>(workspace.liwork));
    workspace.rwork = gpaw::TMalloc<T>(static_cast<size_t>(workspace.lrwork));

    const magma_int_t status = magma_Xheevd<T>(context, inout_eigvecs, inout_eigvals, workspace, false);

    gpaw::TFree(workspace.work);
    gpaw::TFree(workspace.iwork);
    gpaw::TFree(workspace.rwork);

    return interpret_magma_status(status);
}
