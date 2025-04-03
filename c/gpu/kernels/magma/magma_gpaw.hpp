#pragma once

#include "magma_gpaw_interface.h"

#include <magma_auxiliary.h>
#include <magma_types.h>

#ifndef __cplusplus
    #error "C++ needed for GPAW Magma wrappers"
#endif

#include <assert.h>
#include <stdlib.h>

// Check error code of a MAGMA function. Intended for fatal errors, so we exit on failure.
#define MAGMA_CHECK(result) gpawmagma::check(result, __FILE__, __LINE__)

namespace gpawmagma
{

inline void check(magma_int_t result, const char *file, int line)
{
    if (result != MAGMA_SUCCESS) {
        printf("\n\n%s in %s at line %d\n", magma_strerror(result), file, line);
        exit(EXIT_FAILURE);
    }
}

static inline magma_uplo_t get_magma_uplo(char* in_uplo_str)
{
    assert((strcmp(in_uplo_str, "L") == 0 || strcmp(in_uplo_str, "U") == 0)
        && "Invalid UPLO");

    return strcmp(in_uplo_str, "L") == 0 ? MagmaLower : MagmaUpper;
}

// Templated MAGMA malloc on host
template<typename T>
magma_int_t magma_host_malloc(T** ptr, size_t size_in_bytes)
{
    return magma_malloc_cpu(reinterpret_cast<void**>(ptr), size_in_bytes);
}

template<typename T> struct _magma_complex_type;

template<> struct _magma_complex_type<float> { using native_type = magmaFloatComplex; }
template<> struct _magma_complex_type<double> { using native_type = magmaDoubleComplex; }

template<typename T>
using magmaComplex = _magma_complex_type<T>::native_type;

template<typename T>
typedef struct syevd_workgroup
{
    magma_int_t lwork;
    magma_int_t liwork;
    T* work;
    magma_int_t* iwork;
} syevd_workgroup;

template<typename RealT>
typedef struct heevd_workgroup
{
    magma_int_t lwork;
    magma_int_t lrworek;
    magma_int_t liwork;
    magmaComplex<RealT>* work;
    RealT* rwork;
    magma_int_t* iwork;
} heevd_workgroup;

} // namespace gpawmagma
