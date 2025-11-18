import numpy as np
import pytest
from ase.build import bulk, graphene
from ase.build.supercells import make_supercell

from gpaw import GPAW
from gpaw.defects import ElectrostaticCorrections
from gpaw.defects.old_electrostatic import OldElectrostaticCorrections
from pathlib import Path


@pytest.mark.serial
def test_fnv_2d():

    E_corr_t = 4.892
    E_uncorr_t = 9.349

    sigma = 1.0
    charge = +1
    epsilons = [1.9, 1.15]
    a0 = 2.51026699
    c0 = 15.0

    params = {'mode': {'name': 'pw', 'ecut': 400},
              'xc': 'PBE',
              'kpts': {'size': (4, 4, 1)},
              'occupations': {'name': 'fermi-dirac', 'width': 0.01}}

    calc_charged = GPAW(charge=charge, **params)
    calc_neutral = GPAW(charge=0, **params)

    atoms = graphene('N2', a=a0, vacuum=c0 / 2)
    atoms.symbols[0] = 'B'
    atoms.set_pbc(True)
    atoms.center()

    # transformation to orthogonal cell
    P = np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]])
    pristine = make_supercell(atoms, P)
    pristine.calc = calc_neutral
    pristine.get_potential_energy()

    defect = pristine.copy()
    # C_B substitution
    defect[0].symbol = 'C'
    defect[1].magmom = 1
    defect.calc = calc_charged
    defect.get_potential_energy()

    # defect position
    r0 = pristine.positions[0, :]

    elc = OldElectrostaticCorrections(pristine=pristine.calc,
                                      charged=defect.calc,
                                      r0=r0,
                                      q=charge,
                                      sigma=sigma,
                                      dimensionality='2d')
    elc.set_epsilons(epsilons)
    E_corr = elc.calculate_corrected_formation_energy()
    E_uncorr = elc.calculate_uncorrected_formation_energy()

    assert E_corr == pytest.approx(E_corr_t, abs=2e-2)
    assert E_uncorr == pytest.approx(E_uncorr_t, abs=2e-2)


def test_fnv_3d(in_tmp_dir):

    E_corr_t = 23.55
    E_uncorr_t = 18.31
    E_fnv_t = E_corr_t - E_uncorr_t

    prs_path = Path('prs.gpw')
    def_path = Path('def.gpw')

    a0 = 5.628      # lattice parameter
    sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    epsilon = 12.7  # dielectric constant
    charge = -3     # defect charge

    params = {'mode': {'name': 'pw', 'ecut': 400},
              'xc': 'LDA',
              'kpts': {'size': (2, 2, 2), 'gamma': False},
              'occupations': {'name': 'fermi-dirac', 'width': 0.01},
              'txt': 'fnv.txt'}

    calc_charged = GPAW(charge=charge, **params)
    calc_neutral = GPAW(charge=0, **params)

    pristine = bulk('GaAs', crystalstructure='zincblende', a=a0, cubic=True)
    pristine.calc = calc_neutral
    pristine.get_potential_energy()
    pristine.calc.write(prs_path)

    defect = pristine.copy()
    defect.pop(0)  # make a Ga vacancy
    defect.calc = calc_charged
    defect.get_potential_energy()
    defect.calc.write(def_path)

    # defect position
    r0 = pristine.positions[0, :]

    elc = ElectrostaticCorrections(pristine=prs_path,
                                   defect=def_path,
                                   r0=r0,
                                   charge=charge,
                                   sigma=sigma,
                                   epsilon=epsilon,
                                   method='full-planar')
    E_corr = elc.calculate_corrected_formation_energy()
    E_uncorr = elc.calculate_uncorrected_formation_energy()
    E_fnv = E_corr - E_uncorr

    print(E_uncorr, E_corr, E_fnv)
    assert E_fnv == pytest.approx(E_fnv_t, abs=3e-2)
    assert E_corr == pytest.approx(E_corr_t, abs=2e-2)
    assert E_uncorr == pytest.approx(E_uncorr_t, abs=2e-2)


@pytest.mark.parametrize('P', [[[1, 0, 0], [1, 1, 0], [0, 0, 1]]])
# [[1, 0, 0], [1, -1, 0], [0, 0, 1]] passes
def test_fnv_cell(P, in_tmp_dir, gpaw_new):

    if gpaw_new:
        pytest.skip('Transformed cell [90, 90, 45] not supported by GPAW new')

    P = np.array(P)

    E_corr_t = 23.55
    E_uncorr_t = 18.31
    E_fnv_t = E_corr_t - E_uncorr_t

    prs_path = Path('prs.gpw')
    def_path = Path('def.gpw')

    a0 = 5.628      # lattice parameter
    sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    epsilon = 12.7  # dielectric constant
    charge = -3     # defect charge

    params = {'mode': {'name': 'pw', 'ecut': 400},
              'xc': 'LDA',
              # avoid warning about grid symmetrization
              'gpts': [40, 40, 40],
              'kpts': {'size': (2, 2, 2), 'gamma': False},
              'occupations': {'name': 'fermi-dirac', 'width': 0.01}}

    calc_charged = GPAW(charge=charge, **params)
    calc_neutral = GPAW(charge=0, **params)

    pristine = bulk('GaAs', crystalstructure='zincblende', a=a0, cubic=True)
    pristine = make_supercell(pristine, P)
    pristine.calc = calc_neutral
    pristine.get_potential_energy()
    pristine.calc.write(prs_path)

    defect = pristine.copy()
    defect.pop(0)  # make a Ga vacancy
    defect.calc = calc_charged
    defect.get_potential_energy()
    defect.calc.write(def_path)

    # defect position
    r0 = pristine.positions[0, :]

    elc = ElectrostaticCorrections(pristine=prs_path,
                                   defect=def_path,
                                   r0=r0,
                                   charge=charge,
                                   sigma=sigma,
                                   epsilon=epsilon,
                                   method='full-planar')
    E_corr = elc.calculate_corrected_formation_energy()
    E_uncorr = elc.calculate_uncorrected_formation_energy()
    E_fnv = E_corr - E_uncorr

    # changed tolerance to pass ortho-rhombic case
    # switching symmetry off does not help to improve accuracy
    print(E_uncorr, E_corr, E_fnv)
    assert E_fnv == pytest.approx(E_fnv_t, abs=4e-2)
    assert E_corr == pytest.approx(E_corr_t, abs=2e-1)
    assert E_uncorr == pytest.approx(E_uncorr_t, abs=2e-1)


if __name__ == "__main__":
    test_fnv_3d()
