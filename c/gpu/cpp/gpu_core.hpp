#pragma once

#include "gpu/gpu-runtime.h"
#include "utils.hpp"

#include <string>
#include <cstdint>

namespace gpaw
{

inline int32_t get_current_device()
{
    int dev;
    gpuGetDevice(&dev);
    return static_cast<int32_t>(dev);
}

template<typename T>
bool is_pointer_on_device(int32_t device, const T* ptr)
{
    static_assert(!std::is_pointer_v<T>);
    gpuPointerAttributes attr;
    gpuSafeCall(gpuPointerGetAttributes(&attr, reinterpret_cast<const void*>(ptr)));

    return (device == (int32_t)attr.device && attr.type == gpuMemoryTypeDevice);
}

static inline void raise_kernel_launch_error(
    gpuError_t status,
    dim3 blocks,
    dim3 threads,
    size_t shmem_bytes,
    const SourceLocation& original_location
)
{
    printf("Kernel launch error in %s at line %d:\n%s\n",
        original_location.file, original_location.line, gpuGetErrorString(status)
    );
    printf("\nLaunch params were:\n");
    printf("\tblocks (%u, %u, %u)\n", blocks.x, blocks.y, blocks.z);
    printf("\tthreads (%u, %u, %u)\n", threads.x, threads.y, threads.z);
    printf("\tshmem %zu\n", shmem_bytes);
    //printf("\tstream %d\n", static_cast<int>(stream)); // won't work, can't convert streams
    fflush(stdout);
    gpaw_set_runtime_error(gpuGetErrorString(status));
}

// Wrap kernel launch template in a helper struct to make variadic args + default source_location work.
template<typename Kernel, typename... Args>
struct launch_kernel
{
    launch_kernel(
        Kernel kernel,
        dim3 blocks,
        dim3 threads,
        size_t shmem_bytes,
        gpuStream_t stream,
        Args&&... kernel_args,
        SourceLocation source_location = SourceLocation::current()
    )
    {
        static_assert(std::is_invocable_r_v<void, Kernel, Args...>,
            "Incorrect kernel signature, must be f(Args...) -> void"
        );
        gpuLaunchKernel(kernel, blocks, threads, shmem_bytes, stream, std::forward<Args>(kernel_args)...);

        // Check kernel launch.
        // NB: kernel error checking is different in CUDA/HIP, and changes in ROCm 7.0.
        // So this should be revisited at some point.
        gpuError_t status = gpuGetLastError();
        if (status != gpuSuccess)
        {
            raise_kernel_launch_error(status, blocks, threads, shmem_bytes, source_location);
        }
    }
};

// And define a deduction guide for the above. Callable with launch_kernel(Kernel, ...)

/* "Raw" kernel launcher. This launches a given kernel with given arguments
and checks that the launch was OK. On failed launch, a Python runtime error is raised.
Return value is true if the launch was OK, false otherwise. */
template<typename Kernel, typename... Args>
launch_kernel(Kernel, dim3, dim3, size_t, gpuStream_t, Args&&...)
    -> launch_kernel<Kernel, Args...>;


// Schedule an async host callback in the input GPU stream
template <typename F>
void gpu_host_callback(gpuStream_t stream, F&& func)
{
    // Need to wrap the input function in a format that gpuLaunchHostFunc can accept

    using FuncType = std::function<void()>;
    auto *heapFunc = new FuncType(std::forward<F>(func));

    auto trampoline = [](void *data)
    {
        std::unique_ptr<FuncType> fn(static_cast<FuncType*>(data));
        (*fn)();
        // heapFunc deleted when the unique_ptr goes out of scope
    };

    gpuLaunchHostFunc(stream, trampoline, heapFunc);
}

} // namespace gpaw
