import numpy as np
import pytest
from gpaw.new.etdm.skewherm_matrix import SkewHermitian
from scipy.linalg import expm


@pytest.mark.parametrize(
    "ndim,dtype",
    [
        (2, float),
        (2, complex),
        (3, complex),
    ],
)
def test_skewhermitian_rotation(ndim, dtype):
    """
    Test that SkewHermitian rotation_mat correctly computes
    the matrix exponential of a manually reconstructed skew-Hermitian matrix.
    """
    # Step 1: Create a random parameter vector
    if dtype == float:
        vec_len = (ndim * (ndim - 1)) // 2
    else:
        vec_len = (ndim * (ndim + 1)) // 2

    np.random.seed(42)
    param_vec = np.random.random(vec_len).astype(dtype)
    if dtype == complex:
        param_vec += 1j * np.random.random(vec_len)

    print("\n=== Parameter vector ===")
    print(param_vec)

    # Step 2: Manually reconstruct full skew-Hermitian matrix
    ind_up = (
        np.triu_indices(ndim, 1)
        if dtype == float
        else np.triu_indices(ndim)
    )
    a_mat_manual = np.zeros((ndim, ndim), dtype=dtype)
    a_mat_manual[ind_up] = param_vec
    a_mat_manual -= a_mat_manual.T.conj()
    # match vec2skewmat
    np.fill_diagonal(a_mat_manual, a_mat_manual.diagonal() * 0.5)

    print("\n=== Manual skew-Hermitian matrix ===")
    print(a_mat_manual)

    # Step 3: Compute expected matrix exponential manually
    U_manual = expm(a_mat_manual)

    print("\n=== Manual matrix exponential ===")
    print(U_manual)

    # Step 4: Use SkewHermitian class
    sk = SkewHermitian(ndim, dtype, data=param_vec)
    U_class = sk.rotation_mat

    print("\n=== SkewHermitian.rotation_mat ===")
    print(U_class)

    # Step 5: Compare results
    np.testing.assert_allclose(U_class, U_manual, rtol=1e-12, atol=1e-12)

    # Optional: check eigenvalues/eigenvectors consistency
    e_manual, v_manual = np.linalg.eig(1j * a_mat_manual)
    print("\n=== Eigenvalues (manual) ===")
    print(np.sort(e_manual))
    print("=== Eigenvalues (class) ===")
    print(np.sort(sk.evals))
    np.testing.assert_allclose(
        np.sort(e_manual),
        np.sort(sk.evals),
        rtol=1e-12,
        atol=1e-12,
    )


if __name__ == "__main__":
    pytest.main([__file__])
