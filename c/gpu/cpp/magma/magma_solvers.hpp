#pragma once

#include "magma_gpaw.hpp"
#include "magma_template_wrappers.hpp"
#include "gpu/cpp/utils.hpp"

namespace gpaw
{

template<typename T>
magma_int_t magma_symmetric_solver_host(
    const MagmaEighContext& context,
    T* inout_matrix,
    T* inout_eigvals)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");

    SyevdWorkspace<T> workspace;
    bool bQuery = true;
    MAGMA_CHECK(magma_Xsyevd<T>(context, nullptr, nullptr, workspace, bQuery));

    workspace.work = gpaw::TMalloc<T>(static_cast<size_t>(workspace.lwork));
    workspace.iwork = gpaw::TMalloc<magma_int_t>(static_cast<size_t>(workspace.liwork));

    bQuery = false;
    const magma_int_t status = magma_Xsyevd<T>(context, inout_matrix, inout_eigvals, workspace, bQuery);

    gpaw::TFree(workspace.work);
    gpaw::TFree(workspace.iwork);

    return status;
}

template<typename T>
magma_int_t magma_hermitian_solver_host(
    const MagmaEighContext& context,
    magmaComplex<T>* inout_matrix,
    T* inout_eigvals)
{
    // Note that T here is a real type
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");

    // Workspace query
    HeevdWorkspace<T> workspace;
    MAGMA_CHECK(magma_Xheevd<T>(context, nullptr, nullptr, workspace, true));

    workspace.work = gpaw::TMalloc<magmaComplex<T>>(static_cast<size_t>(workspace.lwork));
    workspace.iwork = gpaw::TMalloc<magma_int_t>(static_cast<size_t>(workspace.liwork));
    workspace.rwork = gpaw::TMalloc<T>(static_cast<size_t>(workspace.lrwork));

    const magma_int_t status = magma_Xheevd<T>(context, inout_matrix, inout_eigvals, workspace, false);

    gpaw::TFree(workspace.work);
    gpaw::TFree(workspace.iwork);
    gpaw::TFree(workspace.rwork);

    return status;
}

template<typename T>
magma_int_t magma_symmetric_solver_gpu(
    const MagmaEighContext& context,
    T* inout_matrix,
    T* inout_eigvals)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");

    const size_t n = context.matrix_size;

    // Query
    SyevdWorkspace_gpu<T> workspace;
    MAGMA_CHECK(magma_Xsyevd_gpu<T>(context, nullptr, nullptr, workspace, true));

    // All work buffers are on HOST. Magma also needs the eigenvalue buffer on host
    MAGMA_CHECK(magma_host_malloc(&workspace.work, static_cast<size_t>(workspace.lwork)));
    MAGMA_CHECK(magma_host_malloc(&workspace.iwork, static_cast<size_t>(workspace.liwork)));
    MAGMA_CHECK(magma_host_malloc(&workspace.wA, n * static_cast<size_t>(workspace.ldwa)));

    T* h_eigvals;
    MAGMA_CHECK(magma_host_malloc(&h_eigvals, n));

    const magma_int_t status = magma_Xsyevd_gpu<T>(context, inout_matrix, h_eigvals, workspace, false);

    // Copy eigenvalues back to device
    gpuMemcpy(inout_eigvals, h_eigvals, n*sizeof(T), gpuMemcpyHostToDevice);

    MAGMA_CHECK(magma_host_free(h_eigvals));
    MAGMA_CHECK(magma_host_free(workspace.wA));
    MAGMA_CHECK(magma_host_free(workspace.iwork));
    MAGMA_CHECK(magma_host_free(workspace.work));

    return status;
}

template<typename T>
magma_int_t magma_hermitian_solver_gpu(
    const MagmaEighContext& context,
    magmaComplex<T>* inout_matrix,
    T* inout_eigvals)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");

    const size_t n = context.matrix_size;

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

    const magma_int_t status = magma_Xheevd_gpu<T>(context, inout_matrix, h_eigvals, workspace, false);

    // Copy eigenvalues back to device
    gpuMemcpy(inout_eigvals, h_eigvals, n*sizeof(T), gpuMemcpyHostToDevice);

    MAGMA_CHECK(magma_host_free(h_eigvals));
    MAGMA_CHECK(magma_host_free(workspace.rwork));
    MAGMA_CHECK(magma_host_free(workspace.wA));
    MAGMA_CHECK(magma_host_free(workspace.iwork));
    MAGMA_CHECK(magma_host_free(workspace.work));

    return status;
}

} // namespace gpaw
