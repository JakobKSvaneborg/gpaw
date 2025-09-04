#pragma once

#ifndef __cplusplus
#error "Need C++"
#endif

#include "../../gpaw_utils.h"
#include "../gpu-runtime.h"

#include <functional>
#include <memory>
#include <type_traits>

namespace gpaw
{

// Use in if constexpr (...) to indicate an impossible branch
template<bool flag = false> void static_no_match() { static_assert(flag, "No constexpr match"); }


// Templated C-style malloc. NB: allocs num_elements * sizeof(T) bytes
template<typename T>
T* TMalloc(size_t num_elements)
{
    return static_cast<T*>(malloc(num_elements * sizeof(T)));
}

// Templated C-style free()
template<typename T>
void TFree(T* ptr)
{
    free(static_cast<void*>(ptr));
}

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

#define GPAW_LAUNCH_KERNEL(kernel, blocks, threads, shmem, stream, ...) \
    gpaw::launch_kernel_impl(__FILE__, __LINE__, kernel, blocks, threads, shmem, stream, __VA_ARGS__)

template<typename Kernel, typename... Args>
void launch_kernel_impl(
    const char *file,
    int32_t line,
    Kernel kernel,
    dim3 blocks,
    dim3 threads,
    size_t shmem_bytes,
    gpuStream_t stream,
    Args... args
)
{
    static_assert(std::is_invocable_r_v<void, Kernel, Args...>,
        "Incorrect kernel signature, must be void(Args...)");

    // NB: kernel error checking is different in CUDA/HIP, and changes in ROCm 7.0.
    // So this should be revisited at some point.
    gpuLaunchKernel(kernel, blocks, threads, shmem_bytes, stream, std::forward<Args>(args)...);
    // Check kernel launch
    gpuError_t status = gpuGetLastError();
    if (status != gpuSuccess)
    {
        printf("Kernel launch error in %s at line %d:\n%s\n",
            file, line, gpuGetErrorString(status)
        );
        fflush(stdout);
        gpaw_set_runtime_error(gpuGetErrorString(status));
    }
}

} // namespace gpaw
