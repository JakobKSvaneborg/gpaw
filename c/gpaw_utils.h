#pragma once

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

/* Sets a runtime error to the Python interpreter. Caller should check with PyErr_Occured()
and proceed accordingly. */
CLINKAGE void gpaw_set_runtime_error(const char* err_msg);
// Aborts the program as gracefully as possible. Also raises a Python error
CLINKAGE void gpaw_abort(const char* err_msg);
