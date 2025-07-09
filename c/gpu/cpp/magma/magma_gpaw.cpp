#include "magma_gpaw.hpp"
#include "magma_templates_cpu.hpp"
#include "magma_templates_gpu.hpp"

EighErrorType magma_eigh_cpu(const MagmaEighContext& context, const void* const in_matrix, void* inout_eigvals, void* inout_eigvecs)
{
    assert(in_matrix);
    assert(inout_eigvals);
    assert(inout_eigvecs);

    switch (context.solver_type)
    {
        case EighSolverType::eSsyevd:
            return magma_symmetric_solver_cpu<float>(
                context,
                static_cast<const float*>(in_matrix),
                static_cast<float*>(inout_eigvals),
                static_cast<float*>(inout_eigvecs)
            );

        case EighSolverType::eDsyevd:
            return magma_symmetric_solver_cpu<double>(
                context,
                static_cast<const double*>(in_matrix),
                static_cast<double*>(inout_eigvals),
                static_cast<double*>(inout_eigvecs)
            );

        case EighSolverType::eCheevd:
            return magma_hermitian_solver_cpu<float>(
                context,
                static_cast<const magmaComplex<float>*>(in_matrix),
                static_cast<float*>(inout_eigvals),
                static_cast<magmaComplex<float>*>(inout_eigvecs)
            );

        case EighSolverType::eZheevd:
            return magma_hermitian_solver_cpu<double>(
                context,
                static_cast<const magmaComplex<double>*>(in_matrix),
                static_cast<double*>(inout_eigvals),
                static_cast<magmaComplex<double>*>(inout_eigvecs)
            );

        default:
            break;
    }

    // Should not get here
    return EighErrorType::eInvalidArgument;
}

EighErrorType magma_eigh_gpu(const MagmaEighContext& context, void* inout_matrix, void* inout_eigvals)
{
    assert(inout_matrix);
    assert(inout_eigvals);

    switch (context.solver_type)
    {
        case EighSolverType::eSsyevd:
            return magma_symmetric_solver_gpu<float>(
                context,
                static_cast<float*>(inout_matrix),
                static_cast<float*>(inout_eigvals)
            );

        case EighSolverType::eDsyevd:
            return magma_symmetric_solver_gpu<double>(
                context,
                static_cast<double*>(inout_matrix),
                static_cast<double*>(inout_eigvals)
            );

        case EighSolverType::eCheevd:
            return magma_hermitian_solver_gpu<float>(
                context,
                static_cast<magmaComplex<float>*>(inout_matrix),
                static_cast<float*>(inout_eigvals)
            );

        case EighSolverType::eZheevd:
            return magma_hermitian_solver_gpu<double>(
                context,
                static_cast<magmaComplex<double>*>(inout_matrix),
                static_cast<double*>(inout_eigvals)
            );

        default:
            break;
    }

    return EighErrorType::eInvalidArgument;
}
