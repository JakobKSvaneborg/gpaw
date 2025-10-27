#include "magma_gpaw.hpp"
#include "magma_solvers.hpp"


// Eigensolver that takes host arrays as input and output. Computations are still done using GPU
EighErrorType magma_eigh_host(const MagmaEighContext& context, void* inout_matrix, void* inout_eigvals)
{
    assert(inout_matrix);
    assert(inout_eigvals);

    switch (context.solver_type)
    {
        case EighSolverType::eSsyevd:
            return magma_symmetric_solver_host<float>(
                context,
                static_cast<float*>(inout_matrix),
                static_cast<float*>(inout_eigvals)
            );

        case EighSolverType::eDsyevd:
            return magma_symmetric_solver_host<double>(
                context,
                static_cast<double*>(inout_matrix),
                static_cast<double*>(inout_eigvals)
            );

        case EighSolverType::eCheevd:
            return magma_hermitian_solver_host<float>(
                context,
                static_cast<magmaComplex<float>*>(inout_matrix),
                static_cast<float*>(inout_eigvals)
            );

        case EighSolverType::eZheevd:
            return magma_hermitian_solver_host<double>(
                context,
                static_cast<magmaComplex<double>*>(inout_matrix),
                static_cast<double*>(inout_eigvals)
            );

        default:
            break;
    }

    // Should not get here
    return EighErrorType::eInvalidArgument;
}

// Eigensolver that takes GPU arrays as input and output
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
