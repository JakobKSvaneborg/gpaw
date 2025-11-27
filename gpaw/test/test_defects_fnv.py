import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from gpaw import GPAW
from gpaw.defects import ElectrostaticCorrections
from gpaw.defects.electrostatics import (gather_electrostatic_potential,
                                         build_ugarray, plot_potentials)
from scipy.special import erf


def phi_infty(r_vR, r0_v, Q, alpha):
    """
    Electrostatic potential in atomic units (Hartree) from a Gaussian charge:
        phi(r) = Q * erf(sqrt(alpha)*r) / r
    r in Bohr, phi in Hartree.
    r_vR has shape (3, nx, ny, nz).
    """

    # radius
    r = np.linalg.norm(r_vR - r0_v[:, None, None, None], axis=0)
    phi = np.empty_like(r)

    # mask for nonzero r
    mask = r > 0.0
    sqrt_a = np.sqrt(alpha)

    # normal points
    phi[mask] = Q * erf(sqrt_a * r[mask]) / r[mask]

    # center point: analytic limit
    # lim_{r -> 0} erf(sqrt(a) r)/r = 2 * sqrt(a/pi)
    phi0 = Q * 2.0 * np.sqrt(alpha / np.pi)
    phi[~mask] = phi0

    return phi


def phi_simple(Q, L0, r0, alpha0, ng=64):

    # convert to Bohr
    L = L0 / Bohr
    alpha = alpha0 / Bohr
    r0_v = r0 / Bohr

    x = np.linspace(0, L, ng)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # construct r_vR with shape (3, ng, ng, ng)
    r_vR = np.stack((X, Y, Z), axis=0)

    # potential in eV
    return phi_infty(r_vR, r0_v, Q, alpha) * Hartree


@pytest.mark.parametrize('method', ['atoms', 'full-planar'])
def test_fnv_model(method):

    L = 20.0
    epsilon = 1.0
    charge = -2.0
    sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0))) * Bohr
    E_fnv_t = {'atoms': 6.56, 'full-planar': 7.62}

    L2 = L / 2
    L0 = L / 8
    pos = [[L2, L2, L2], [L0, L0, L0]]
    pristine = Atoms('H2', positions=pos, cell=[L, L, L])
    pristine.set_pbc(True)

    # defect position
    r0 = pristine.positions[0, :]

    phi_def = phi_simple(Q=charge, L0=L, r0=r0, alpha0=1.0)
    phi_def_R = build_ugarray(pristine, phi_def)

    phi_prs = np.zeros_like(phi_def)
    phi_prs_R = build_ugarray(pristine, phi_prs)

    elc = ElectrostaticCorrections(phi_pristine=phi_prs_R,
                                   phi_defect=phi_def_R,
                                   r0=r0,
                                   charge=charge,
                                   sigma=sigma,
                                   epsilon=epsilon,
                                   method=method,
                                   atoms_pristine=pristine)
    E_fnv = elc.calculate_correction()

    if 0:
        profile = elc.calculate_potential_profile()
        plot_potentials(profile)

    assert E_fnv == pytest.approx(E_fnv_t[method], abs=1e-2)


@pytest.mark.parametrize('cell', ['cubic', 'skew'])
def test_fnv_3d(gpw_files, cell):

    E_corr_t = 23.55
    E_uncorr_t = 18.31
    E_fnv_t = E_corr_t - E_uncorr_t

    if cell == 'cubic':
        tol = 3e-2
    elif cell == 'skew':
        tol = 5e-2

    sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0))) * Bohr
    epsilon = 12.7  # dielectric constant
    charge = -3     # defect charge
    calc_prs = GPAW(gpw_files[f'gaas_{cell}_pristine'])
    calc_def = GPAW(gpw_files[f'gaas_{cell}_defect'])

    atoms = calc_prs.get_atoms()
    phiR_prs = gather_electrostatic_potential(calc_prs)
    phiR_def = gather_electrostatic_potential(calc_def)

    # defect position
    r0 = atoms.positions[0, :]

    elc = ElectrostaticCorrections(phi_pristine=phiR_prs,
                                   phi_defect=phiR_def,
                                   r0=r0,
                                   charge=charge,
                                   sigma=sigma,
                                   epsilon=epsilon,
                                   method='full-planar',
                                   atoms_pristine=atoms)
    E_fnv = elc.calculate_correction()

    if 0:
        profile = elc.calculate_potential_profile()
        plot_potentials(profile)

    assert E_fnv == pytest.approx(E_fnv_t, abs=tol)


if __name__ == "__main__":
    test_fnv_model()
