#pragma once

#include "magma_python_interface.h"

#include <magma_auxiliary.h>
#include <magma_types.h>

#ifndef __cplusplus
    #error "C++ needed for GPAW Magma wrappers"
#endif

#include "../utils.hpp"

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cstdio>

// Check error code of a MAGMA function. Intended for fatal errors, so we exit on failure.
#define MAGMA_CHECK(result) gpaw_magma_errcheck(result, __FILE__, __LINE__)

inline void gpaw_magma_errcheck(magma_int_t result, const char *file, int line)
{
    if (result != MAGMA_SUCCESS) {
        printf("\n\n%s in %s at line %d\n", magma_strerror(result), file, line);
        exit(EXIT_FAILURE);
    }
}


// Templated MAGMA malloc on host, allocs num_elements * sizeof(T) bytes
template<typename T>
inline magma_int_t magma_host_malloc(T** ptr, size_t num_elements)
{
    return magma_malloc_cpu(reinterpret_cast<void**>(ptr), num_elements * sizeof(T));
}

// Templated MAGMA free on host
template<typename T>
inline magma_int_t magma_host_free(T* ptr)
{
    return magma_free_cpu(static_cast<void*>(ptr));
}

template<typename T> struct _magma_complex_type;
template<> struct _magma_complex_type<float> { using native_type = magmaFloatComplex; };
template<> struct _magma_complex_type<double> { using native_type = magmaDoubleComplex; };

// Templated Magma complex number
template<typename T>
using magmaComplex = typename _magma_complex_type<T>::native_type;


enum class EighSolverType : uint8_t
{
    eNone,              // invalid
    eSsyevd,            // single precision real symmetric
    eDsyevd,            // double precision real symmetric
    eCheevd,            // single precision complex Hermitian
    eZheevd,            // double precision complex Hermitian
};


// Info about the input matrix and on what the solver should do
struct MagmaEighContext
{
    EighSolverType solver_type;
    // Do eigenvectors?
    magma_vec_t jobz;
    magma_uplo_t uplo;
    magma_int_t matrix_size;
    magma_int_t matrix_lda;
};

enum class EighErrorType
{
    eSuccess,
    eInvalidArgument,
    eFailedToConverge
};

template<typename T>
struct SyevdWorkspace
{
    magma_int_t lwork;
    magma_int_t liwork;
    // work buffers, always on host
    T* work;
    magma_int_t* iwork;
};

template<typename RealT>
struct HeevdWorkspace
{
    magma_int_t lwork;
    magma_int_t lrwork;
    magma_int_t liwork;
    // work buffers, always on host
    magmaComplex<RealT>* work;
    RealT* rwork;
    magma_int_t* iwork;
};

// GPU solvers need an additional work buffer

template<typename T>
struct SyevdWorkspace_gpu : public SyevdWorkspace<T>
{
    // Dimension (LDWA, N)
    T* wA;
    magma_int_t ldwa = 0;
};

template<typename T>
struct HeevdWorkspace_gpu : public HeevdWorkspace<T>
{
    // Dimension (LDWA, N)
    magmaComplex<T>* wA;
    magma_int_t ldwa = 0;
};


inline EighErrorType interpret_magma_status(magma_int_t status)
{
    if (status > 0)
    {
        return EighErrorType::eFailedToConverge;
    }
    else if (status < 0)
    {
        return EighErrorType::eInvalidArgument;
    }
    else
    {
        return EighErrorType::eSuccess;
    }
}

// We do manual type erasure to implement polymorphic entry points for the solvers (ie. inputs are void*).
// Functions called from Python operate on Python array objects and pass their data pointers to type-erased solvers.
// Inside the entry functions we cast back to the correct types.

/* Entry point to Magma CPU eigensolvers.
* The pointers must point to accessible memory locations of correct size.
* The input/output matrices are in Magma conventions, NOT in Numpy/Python style conventions.
*/
EighErrorType magma_eigh_cpu(const MagmaEighContext& context, const void* const in_matrix, void* inout_eigvals, void* inout_eigvecs);


/* Entry point to Magma single-GPU eigensolvers.
* The pointers must point to accessible memory on the device.
* The input/output matrices are in Magma conventions, NOT in Numpy/Python style conventions.
*/
EighErrorType magma_eigh_gpu(const MagmaEighContext& context, const void* const in_matrix, void* inout_eigvals, void* inout_eigvecs);
