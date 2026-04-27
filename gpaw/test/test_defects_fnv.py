import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from gpaw.defects import (ElectrostaticCorrections,
                          charged_defect_corrections)
from gpaw.defects.electrostatics import build_ugarray, plot_potentials


def phi_gaussian(Q, L0, r0, alpha0, ng=32):
    """
    Electrostatic potential in eV
    Q  charge
    L0 box size [Angstrom]
    r0 position [Angstrom]
    alpha0 extend of gaussian [Angstrom]
    ng grid size
    """

    # convert to Bohr
    L = L0 / Bohr
    alpha = alpha0 / Bohr
    r0_v = r0 / Bohr

    x = np.linspace(0, L, ng)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # construct r_vR with shape (3, ng, ng, ng)
    r_vR = np.stack((X, Y, Z), axis=0)

    # radius
    r = np.linalg.norm(r_vR - r0_v[:, None, None, None], axis=0)
    phi = np.zeros_like(r)
    rmax = 3 * alpha
    mask = r < rmax
    phi[mask] = np.exp(- (r[mask] / alpha)**2) / alpha

    # potential in eV
    return - Q * phi / np.sqrt(2. * np.pi) * Hartree


@pytest.mark.parametrize('method', ['atoms', 'sparse-planar'])
def test_fnv_model(method):

    L = 15.0
    epsilon = 5.0
    charge = -2.0
    alpha = 1.0
    sigma = 1.5
    E_fnv_t = {'atoms': 0.47, 'sparse-planar': 0.84}

    L2 = L / 2
    L0 = 0.0
    pos = [[L2, L2, L2], [L0, L0, L0]]
    pristine = Atoms('H2', positions=pos, cell=[L, L, L])
    pristine.set_pbc(True)

    # defect position
    r0 = pristine.positions[0, :]

    phi_def = phi_gaussian(Q=charge, L0=L, r0=r0, alpha0=alpha)
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
                                   ravg=alpha,
                                   atoms_pristine=pristine)
    E_fnv = elc.calculate_correction()

    if 0:
        profile = elc.calculate_potential_profile()
        plot_potentials(profile)

    assert E_fnv == pytest.approx(E_fnv_t[method], abs=1e-2)


@pytest.mark.parametrize('cell', ['cubic', 'skew'])
def test_fnv_3d(gpw_files, cell, mpi):

    E_corr_t = 23.55
    E_uncorr_t = 18.31
    E_fnv_t = E_corr_t - E_uncorr_t

    if cell == 'cubic':
        tol = 3e-2
    elif cell == 'skew':
        tol = 5e-2

    epsilon = 12.7  # dielectric constant
    charge = -3     # defect charge
    def_idx = 0     # defect index in pristine system
    calc_prs = mpi.GPAW(gpw_files[f'gaas_{cell}_pristine'])
    calc_def = mpi.GPAW(gpw_files[f'gaas_{cell}_defect'])

    elc = charged_defect_corrections(calc_pristine=calc_prs,
                                     calc_defect=calc_def,
                                     defect_index=def_idx,
                                     charge=charge,
                                     epsilon=epsilon)
    E_fnv = elc.calculate_correction()

    if 0:
        profile = elc.calculate_potential_profile()
        plot_potentials(profile)

    assert E_fnv == pytest.approx(E_fnv_t, abs=tol)


if __name__ == "__main__":
    test_fnv_model()
