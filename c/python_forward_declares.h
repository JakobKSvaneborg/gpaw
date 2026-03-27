#pragma once

// We fall back to including full python headers if pytypedefs.h is not found (python <= 3.10)
#if defined(__has_include)
    #if __has_include(<pytypedefs.h>)
        #include <pytypedefs.h>
    #else
        #include "python_utils.h"
    #endif
#else
    #include "python_utils.h"
#endif
