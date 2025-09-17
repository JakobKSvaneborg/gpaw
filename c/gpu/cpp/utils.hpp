#pragma once

#ifndef __cplusplus
#error "Need C++"
#endif

#include "../../gpaw_utils.h"

#include <functional>
#include <memory>
#include <type_traits>
#include <cstdint>
#include <sstream>
#include <unordered_map>

namespace gpaw
{

/* Like std::source_location but works in C++17. Uses Clang/GCC extensions.
TODO: replace with std::source_location when using C++20. */
struct SourceLocation
{

    constexpr SourceLocation(
        const char* file_ = __builtin_FILE(),
        const char* func_ = __builtin_FUNCTION(),
        int line_ = __builtin_LINE()
    ) noexcept : file(file_), function(func_), line(line_)
    {
    }

    static constexpr SourceLocation current(
        const char* file_ = __builtin_FILE(),
        const char* func_ = __builtin_FUNCTION(),
        int line_ = __builtin_LINE()) noexcept
    {
        return SourceLocation(file_, func_, line_);
    }

    // Returns string `filename:line`
    std::string to_string() const
    {
        return std::string(file) + ":" + std::to_string(line);
    }

    const char* file;
    const char* function;
    uint32_t line;
};


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

// Checks if an std::unordered_map contains a key
template<typename Key, typename Val>
inline bool map_contains(const std::unordered_map<Key, Val>& map, Key key)
{
    return map.find(key) != map.end();
}

} // namespace gpaw
