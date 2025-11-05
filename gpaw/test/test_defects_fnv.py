import pytest
import numpy as np
from ase.build import bulk
from gpaw import GPAW
from gpaw.defects import ElectrostaticCorrections


@pytest.mark.parametrize('modename', ['pw', 'fd'])
def test_fnv(modename):
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
    test_fnv('pw')
