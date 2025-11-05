import numpy as np
from ase.build import bulk
from gpaw import GPAW
from gpaw.defects import ElectrostaticCorrections
import pytest
from pathlib import Path


def test_fnv():
    a0 = 5.628  # lattice parameter
    sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    epsilon = 12.7 # dielectric constant
    N = 1
    charge = -3  # defect charge

    gpw_def = Path('GaAs.def.gpw')
    gpw_prs = Path('GaAs.prs.gpw')

    params = {'mode': {'name': 'fd'},
              'xc': 'LDA',
              'kpts': {'size': (2, 2, 2), 'gamma': False},
              'occupations': {'name': 'fermi-dirac', 'width': 0.01}}
              
    E_corr_t = 23.558833
    E_uncorr_t = 18.310214

    pristine = bulk('GaAs', crystalstructure='zincblende', a=a0, cubic=True)

    defect = pristine.copy()
    defect.pop(0)  # make a Ga vacancy

    calc_charged = GPAW(charge=charge, **params) 
    calc_neutral = GPAW(charge=0, **params) 

    defect.calc = calc_charged
    defect.get_potential_energy()
    defect.calc.write(gpw_def)

    # pristine case

    pristine.calc = calc_neutral
    pristine.get_potential_energy()
    pristine.calc.write(gpw_prs)

    # need to convert Path -> str
    elc = ElectrostaticCorrections(pristine=str(gpw_prs),
                                   charged=str(gpw_def),
                                   q=charge,
                                   sigma=sigma,
                                   dimensionality='3d')
    elc.set_epsilons(epsilon)
    E_corr = elc.calculate_corrected_formation_energy()
    E_uncorr = elc.calculate_uncorrected_formation_energy()

    assert E_corr == pytest.approx(E_corr_t, abs=1e-3)
    assert E_uncorr == pytest.approx(E_uncorr_t, abs=1e-3)


if __name__ == "__main__":
    test_fnv()
