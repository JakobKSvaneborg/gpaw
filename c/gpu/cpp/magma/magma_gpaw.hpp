#pragma once

// MAGMA needs stdbool.h but it is not properly included by their own headers.
// Can remove this include once it's fixed in MAGMA.
// See https://github.com/icl-utk-edu/magma/pull/41
#include <stdbool.h>
#include <magma_v2.h>

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


// Check error code of a MAGMA function and throw std::runtime_error on failure
#define MAGMA_CHECK(result) gpaw::gpaw_magma_errcheck(result, __FILE__, __LINE__)

namespace gpaw
{

inline void gpaw_magma_errcheck(magma_int_t result, const char *file, int line)
{
    if (result != MAGMA_SUCCESS)
    {
        std::string msg = "MAGMA error " + std::string(magma_strerror(result))
            + " at " + std::string(file) +  ":" + std::to_string(line);
        throw std::runtime_error(msg);
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

// Info about the input matrix and on what the solver should do
struct MagmaEighContext
{
    // Do eigenvectors?
    magma_vec_t jobz;
    magma_uplo_t uplo;
    magma_int_t matrix_size;
    magma_int_t matrix_lda;
    // How many GPUs to use. Only for the version that has input/output on HOST
    magma_int_t num_gpus;
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

} // namespace gpaw
