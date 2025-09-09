#pragma once

#ifdef __cplusplus
    #define CLINKAGE extern "C"
#else
    #define CLINKAGE
#endif

CLINKAGE void gpaw_set_runtime_error(const char* err_msg);
