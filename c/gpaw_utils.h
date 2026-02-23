#pragma once

// Build system must define GPAW_CPP if compiling all of GPAW as C++

// This check should work for Cray/Intel/NVCC/HIPCC compilers too
#if defined(__GNUC__) || defined(__clang__)
    #define GPAW_HIDDEN_SYMBOL __attribute__((visibility("hidden")))
#else
    // Either MSVC (symbols hidden by default) or unknown compiler (may give compiler warnings about symbol visibility mismatches)
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
supported, otherwise simply leave GPAW_RESTRICT undefined.
Note that #if defined(__restrict__) does not work as __restrict__ is not a macro.*/

#ifdef __cplusplus
    #if defined(__GNUC__) || defined(__clang__)
        #define GPAW_RESTRICT __restrict__
    #elif defined(_MSC_VER)
        #define GPAW_RESTRICT __restrict
    #else
        // Unsupported compiler? Leave empty
        #define GPAW_RESTRICT
        #warning "Could not find C++ compiler extension for 'restrict' keyword. This could be detrimental for performance.\n" \
                 "Please report this to GPAW developers if you are sure your compiler supports some version of `restrict`."
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
