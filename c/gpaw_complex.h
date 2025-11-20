#pragma once

#ifdef __cplusplus
    #include <complex>
    
    // Some headers (lfc.h) are used from both C and C++ code => need a common typedef for complex numbers that works in both C and C++.
    using double_complex = std::complex<double>;

    // Get real part of a complex double. Provided for compatibility with C99 code
    constexpr inline double creal(const double_complex& z) { return z.real(); }

    // Get imaginary part of a complex double. Provided for compatibility with C99 code
    constexpr inline double cimag(const double_complex& z) { return z.imag(); }

    // Computes the complex base-e exponential of z. Provided for compatibility with C99 code
    inline double_complex cexp(const double_complex& z) { return std::exp(z); }
    // NB: std::exp is not constexpr until C++26

    /* DO NOT USE IN NEW CODE!
    Global shorthand for the imaginary unit, needed for compatibility with legacy C code. */
    constexpr double_complex I(0.0, 1.0);

#else
    // C99
    #include <complex.h>

    /* Use double_complex in code instead of "double complex". Will make conversion to C++ easier.
    Do NOT use "complex double", that is a GCC extension. */
    typedef double complex double_complex;
#endif
