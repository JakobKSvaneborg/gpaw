#ifndef MAGMA_GPAW_INTERFACE_H
#define MAGMA_GPAW_INTERFACE_H

// C99 compliant header that can safely be included from main GPAW.
// In particular, gives magma_init() and magma_finalize()

// MAGMA needs stdbool.h but it is not properly included by their own headers.
// Can remove this include once it's fixed in MAGMA.
// See https://github.com/icl-utk-edu/magma/pull/41
#include <stdbool.h>
#include <magma_v2.h>

#endif
