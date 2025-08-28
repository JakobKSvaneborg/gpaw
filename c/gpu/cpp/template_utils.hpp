#pragma once

#ifndef __cplusplus
#error "Need C++"
#endif

#include "../gpu-runtime.h"
#include <functional>
#include <memory>

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
    // Need to wrap an arbitrary function/lambda in a format that gpuLaunchHostFunc can accept

    using FuncType = std::function<void()>;
    auto *heapFunc = new FuncType(std::forward<F>(func));

    auto trampoline = [](void *data)
    {
        std::unique_ptr<FuncType> fn(static_cast<FuncType*>(data));
        (*fn)();
    };

    gpuLaunchHostFunc(stream, trampoline, heapFunc);
}

} // namespace gpaw
