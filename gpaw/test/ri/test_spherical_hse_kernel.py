"""

Test the implementation of spherical harmonic expansion of screened
Coulomb kernel. Based on

    János G Ángyán et al 2006 J. Phys. A: Math. Gen. 39 8613


"""


from gpaw.xc.ri.spherical_hse_kernel import Phi as phi
from scipy.special import erfc
import numpy as np
from gpaw.sphere.lebedev import weight_n, Y_nL, R_nv
from gpaw.sphere.spherical_harmonics import Y


# [23, 31, 16] are indices to a triangle in the 50-point Lebedev grid.
# This is used to align the two angular grids nicely to avoid divergences.
Rdir_v = np.mean(R_nv[[23, 31, 16], :], axis=0)
# Obtain the tangent and bitangent vectors for a full basis
Q_vv, _ = np.linalg.qr(np.array([Rdir_v]).T, 'complete')

# Get the rotated second angular integration grid
R2_nv = R_nv @ Q_vv


def phiold(n, mu, R, r):
    """

        Explicit implementation of spherical harmonic expansion up to l=2
        as given by the article Eqs. A1, A2 and A3. These are compared to
        the official implementation in `xc/ri/spherical_hse_kernel.py`.

    """
    Rg = np.maximum.reduce([R, r])
    Rl = np.minimum.reduce([R, r])
    Xi = mu * Rg
    xi = mu * Rl

    if n == 0:
        prefactor = -1 / (2 * np.pi**0.5 * xi * Xi)
        A = np.exp(-(xi + Xi)**2) - np.exp(-(xi - Xi)**2)
        B = -np.pi**0.5 * ((xi - Xi) * erfc(Xi - xi) +
                           (Xi + xi) * erfc(Xi + xi))
        return mu * prefactor * (A + B)
    if n == 1:
        prefactor = -1 / (2 * np.pi**0.5 * xi**2 * Xi**2)
        A = 1 / 2 * ((np.exp(-(xi + Xi)**2) - np.exp(-(xi - Xi)**2)) * (2 * xi**2 + 2 * xi * Xi - (1 - 2 * Xi**2)) - 4 * xi * Xi * np.exp(-(xi + Xi)**2)) - np.pi**0.5 * ((xi**3 - Xi**3) * erfc(Xi - xi) + (xi**3 + Xi**3) * erfc(xi + Xi))  # noqa: E501
        return mu * prefactor * A
    if n == 2:
        prefactor = -1 / (2 * np.pi**0.5 * xi**3 * Xi**3)
        A = 1 / 4 * ((np.exp(-(xi + Xi)**2) - np.exp(-(xi - Xi)**2)) * (4 * (xi**4 + xi**3 * Xi + Xi**4) - 2 * xi**2 * (1 - 2 * Xi**2) + (1 - 2 * xi * Xi) * (3 - 2 * Xi**2)) - 4 * np.exp(-(xi + Xi)**2) * xi * Xi * (2 * xi**2 - (3 - 2 * Xi**2))) - np.pi**0.5 * ((xi**5 - Xi**5) * erfc(Xi - xi) + (xi**5 + Xi**5) * erfc(xi + Xi))  # noqa: E501
        return mu * prefactor * A
    raise NotImplementedError


def phi_lebedev(n, mu, R_x, r_x):
    # Target spherical harmonic, primary grid
    Y1_n = Y_nL[:, n**2]
    # Target spherical harmonic, secondary grid
    Y2_n = Y(n**2, *R2_nv.T)

    V_x = np.zeros_like(R_x)
    for x, (R, r) in enumerate(zip(R_x, r_x)):
        C1_nv = R * R_nv
        C2_nv = r * R2_nv

        D_nn = np.sum((C1_nv[:, None, :] - C2_nv[None, :, :])**2, axis=2)**0.5
        V_nn = erfc(D_nn * mu) / D_nn

        V_x[x] = np.einsum('n,m,nm,n,m', weight_n, weight_n, V_nn, Y1_n, Y2_n)

    return V_x * (4 * np.pi) * (2 * n + 1)


def test_old_vs_new_spherical_kernel():
    """Test explicitely hard coded implementation against generic
    implementation.
    """
    rng = np.random.RandomState(42)
    for n in range(3):
        R = rng.random(100) * 10
        r = rng.random(100) * 10
        params = (n, 0.11, R, r)
        new, old = phi(*params), phiold(*params)
        print(old, new, abs(old - new).max())

        assert np.allclose(new, old, atol=1.5e-6)


def test_wrt_lebedev_integrated_kernel(plot=False):
    """
        Test a double angular numerically integrated kernel against
        the generic implementation.
    """
    if plot:
        import matplotlib.pyplot as plt
    s = 25
    for n in range(5):
        for RR in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]:
            R = RR * np.ones((125,))
            r = np.logspace(-5, 3, 125 + 1)[1:]
            params = (n, 0.11, R.ravel(), r.ravel())
            new, old = phi(*params), phi_lebedev(*params)
            if plot:
                plt.loglog(r, np.abs(old), '-r')
                plt.loglog(r, np.abs(new), '--b')
                plt.loglog(r, np.abs(old - new), '-k')
                plt.ylim([1e-7, 1e7])
            err = np.abs(new - old)

            # The angular integration error (due to only 50 point grid) is
            # too large on small separations. Therefore, they should not
            # be compared directly.
            err = np.where(R - r < 0.3, 0 * err, err)
            assert np.allclose(err, 0, atol=1e-2, rtol=1e-2)
        if plot:
            plt.show()

    if plot:
        import matplotlib.pyplot as plt
        for n in range(5):
            t = np.logspace(-5, 5, s)
            R, r = np.meshgrid(t, t)
            params = (n, 0.11, R.ravel(), r.ravel())
            new, old = phi(*params), phi_lebedev(*params)
            plt.contourf(np.log10(r), np.log10(R),
                         np.reshape(np.log10(np.abs(old - new) + 1e-7),
                                    (s, s)))
            plt.colorbar()
            plt.show()


if __name__ == '__main__':
    test_wrt_lebedev_integrated_kernel(plot=True)
