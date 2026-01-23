#pragma once

// Build system must define GPAW_CPP if compiling all of GPAW as C++

#ifndef _MSC_VER
    #define GPAW_HIDDEN_SYMBOL __attribute__((visibility("hidden")))
#else
    // MSVC hides symbols by default
    #define GPAW_HIDDEN_SYMBOL
#endif

#ifdef __cplusplus
    #define CLINKAGE extern "C"
    // Starts an extern "C" block
    #define CLINKAGE_BEGIN CLINKAGE {
    // Ends an extern "C" block
    #define CLINKAGE_END   }
#else
    #define CLINKAGE
    #define CLINKAGE_BEGIN
    #define CLINKAGE_END
#endif

/* Handle `restrict` keyword not existing in C++. Use compiler extension if
supported, otherwise simply leave GPAW_RESTRICT undefined */

#ifdef GPAW_CPP
    #if defined(__clang__) || defined(__GNUC__)
        #define GPAW_RESTRICT __restrict__
    #elif defined(_MSC_VER)
        #define GPAW_RESTRICT __restrict
    #elif defined(_CRAYC)
        #define GPAW_RESTRICT __restrict
    #elif defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
        #define GPAW_RESTRICT __restrict
    #else
        // Unsupported compiler? Leave empty
        #define GPAW_RESTRICT
    #endif

#else
    // C99 mode
    #define GPAW_RESTRICT restrict
#endif

// FIXME: remove CLINKAGE from these when removing the C-compilation path

/* Sets a runtime error to the Python interpreter. Caller should check with PyErr_Occured()
and proceed accordingly. */
CLINKAGE void gpaw_set_runtime_error(const char* err_msg);
// Aborts the program as gracefully as possible. Also raises a Python error
CLINKAGE void gpaw_abort(const char* err_msg);
