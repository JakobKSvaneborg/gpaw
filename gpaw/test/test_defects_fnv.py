import pytest
import numpy as np
from ase.build import bulk, graphene
from ase.build.supercells import make_supercell
from ase import Atoms
from gpaw import GPAW
from gpaw.defects import ElectrostaticCorrections

def test_fnv_2d():

    E_corr_t = 4.892
    E_uncorr_t = 9.349

    sigma = 1.0
    charge = +1
    epsilons = [1.9, 1.15]
    a0 = 2.51026699
    c0 = 15.0

    params = {'mode': 'fd',
              'xc': 'PBE',
              'kpts': {'size': (4, 4, 1)},
              'occupations': {'name': 'fermi-dirac', 'width': 0.01}}

    calc_charged = GPAW(charge=charge, **params)
    calc_neutral = GPAW(charge=charge, **params)

    atoms = graphene('N2', a=a0, vacuum=c0/2)
    atoms.symbols[0] = 'B'
    atoms.set_pbc(True)
    atoms.center()
    
    # transformation to orthogonal cell
    P = np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]])

    pristine = make_supercell(atoms, P)
    pristine.edit()
    pristine.calc = calc_neutral
    pristine.get_potential_energy()

    defect = pristine.copy()
    # C_B substitution
    defect[0].symbol = 'C'
    defect[1].magmom = 1
    defect.calc = calc_charged
    defect.get_potential_energy()

    elc = ElectrostaticCorrections(pristine=pristine.calc,
                                   charged=defect.calc,
                                   q=charge,
                                   sigma=sigma,
                                   dimensionality='2d')
    elc.set_epsilons(epsilons)
    E_corr = elc.calculate_corrected_formation_energy()
    E_uncorr = elc.calculate_uncorrected_formation_energy()

    assert E_corr == pytest.approx(E_corr_t, abs=2e-2)
    assert E_uncorr == pytest.approx(E_uncorr_t, abs=2e-2)


@pytest.mark.parametrize('modename', ['pw', 'fd'])
def test_fnv_3d(modename):
    a0 = 5.628      # lattice parameter
    sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    epsilon = 12.7  # dielectric constant
    charge = -3     # defect charge

    mode = {'pw': {'name': 'pw', 'ecut': 400},
            'fd': {'name': 'fd'}}

    params = {'mode': mode[modename],
              'xc': 'LDA',
              'kpts': {'size': (2, 2, 2), 'gamma': False},
              'occupations': {'name': 'fermi-dirac', 'width': 0.01}}

    E_corr_t = 23.55
    E_uncorr_t = 18.31

    pristine = bulk('GaAs', crystalstructure='zincblende', a=a0, cubic=True)

    defect = pristine.copy()
    defect.pop(0)  # make a Ga vacancy

    calc_charged = GPAW(charge=charge, **params)
    calc_neutral = GPAW(charge=0, **params)

    defect.calc = calc_charged
    defect.get_potential_energy()

    # pristine case
    pristine.calc = calc_neutral
    pristine.get_potential_energy()

    # need to convert Path -> str
    elc = ElectrostaticCorrections(pristine=pristine.calc,
                                   charged=defect.calc,
                                   q=charge,
                                   sigma=sigma,
                                   dimensionality='3d')
    elc.set_epsilons(epsilon)
    E_corr = elc.calculate_corrected_formation_energy()
    E_uncorr = elc.calculate_uncorrected_formation_energy()

    assert E_corr == pytest.approx(E_corr_t, abs=2e-2)
    assert E_uncorr == pytest.approx(E_uncorr_t, abs=2e-2)


if __name__ == "__main__":
    test_fnv_2d()
