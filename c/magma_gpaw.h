#ifndef MAGMA_GPAW_H
#define MAGMA_GPAW_H

// MAGMA needs stdbool.h but it is not properly included by their own headers.
// Can remove this include once it's fixed in MAGMA.
// See https://github.com/icl-utk-edu/magma/pull/41
#include <stdbool.h>
#include <assert.h>

#include <magma_v2.h>
#include <magma_auxiliary.h>
#include <magma_types.h>


static inline magma_uplo_t get_magma_uplo(char* in_uplo_str)
{
    assert((strcmp(in_uplo_str, "L") == 0 || strcmp(in_uplo_str, "U") == 0)
        && "Invalid UPLO");

    return strcmp(in_uplo_str, "L") == 0 ? MagmaLower : MagmaUpper;
}

typedef struct dsyevd_workgroup
{
    magma_int_t lwork;
    magma_int_t liwork;
    double* work;
    magma_int_t* iwork;
} dsyevd_workgroup;

typedef struct zheevd_workgroup
{
    magma_int_t lwork;
    magma_int_t lrwork;
    magma_int_t liwork;
    magmaDoubleComplex* work;
    double* rwork;
    magma_int_t* iwork;
} zheevd_workgroup;

#endif
