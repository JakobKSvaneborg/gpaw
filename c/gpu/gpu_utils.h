#pragma once

// TODO: combine with other gpu headers or remove when moving to full C++

/* Some C++ functions in the GPU code need to interface with the main GPAW extension.
* Old builds: Main GPAW is C code, GPU code is C++ => GPU functions called from GPAW extension need C-linkage
* New builds: All of GPAW is C++ => No need for C-linkage in GPU code either.
* To handle both options we define GPAW_GPU_LINKAGE that sets the correct linkage for GPU functions.
* No need to specify the linkage again when defining the function, assuming this header gets included in the file that defines it.
*
* TODO: When removing the old C build path, simply remove the definition of GPAW_GPU_LINKAGE.
*/

#ifdef GPAW_CPP
    // Everything is C++, no need for C-linkage anywhere
    #define GPAW_GPU_LINKAGE
    #define GPAW_GPU_LINKAGE_BEGIN
    #define GPAW_GPU_LINKAGE_END
#else
    // Main GPAW is C-code, need C-linkage for GPU functions that are called from it
    #define GPAW_GPU_LINKAGE        CLINKAGE
    #define GPAW_GPU_LINKAGE_BEGIN  CLINKAGE_BEGIN
    #define GPAW_GPU_LINKAGE_END    CLINKAGE_END
#endif

GPAW_GPU_LINKAGE void gpaw_device_synchronize();
