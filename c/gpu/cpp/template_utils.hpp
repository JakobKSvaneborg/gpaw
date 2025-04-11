#pragma once

#ifndef __cplusplus
#error "Need C++"
#endif

namespace gpaw
{

// Common templates for better type safety when interfacing with C stdlib

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

// Use in if constexpr (...) to indicate an impossible branch
template<bool flag = false> void static_no_match() { static_assert(flag, "No constexpr match"); }

} // namespace gpaw
