import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.defects import ElectrostaticCorrections
import pytest


def test_fnv():
    a0 = 5.628  # Lattice parameter
    sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    epsilon = 12.7
    N = 1
    q = -3  # Defect charge

    E_corr_t = 23.558833
    E_uncorr_t = 18.310214

    formula = 'Ga4As4'

    lattice = [[a0, 0.0, 0.0],  # work with cubic cell
               [0.0, a0, 0.0],
               [0.0, 0.0, a0]]

    basis = [[0.0, 0.0, 0.0],
             [0.5, 0.5, 0.0],
             [0.0, 0.5, 0.5],
             [0.5, 0.0, 0.5],
             [0.25, 0.25, 0.25],
             [0.75, 0.75, 0.25],
             [0.25, 0.75, 0.75],
             [0.75, 0.25, 0.75]]

    GaAs = Atoms(symbols=formula,
                 scaled_positions=basis,
                 cell=lattice,
                 pbc=(1, 1, 1))

    GaAsdef = GaAs.repeat((N, N, N))

    GaAsdef.pop(0)  # Make the supercell and a Ga vacancy

    calc = GPAW(mode='fd',
                kpts={'size': (2, 2, 2), 'gamma': False},
                xc='LDA',
                charge=q,
                occupations={'name': 'fermi-dirac', 'width': 0.01},
                txt='GaAs{0}{0}{0}.Ga_vac.txt'.format(N))

    GaAsdef.calc = calc
    GaAsdef.get_potential_energy()

    calc.write('GaAs{0}{0}{0}.Ga_vac_charged.gpw'.format(N))

    # now for the pristine case

    GaAspris = GaAs.repeat((N, N, N))
    parameters = calc.todict()
    parameters['txt'] = 'GaAs{0}{0}{0}.pristine.txt'.format(N)
    parameters['charge'] = 0
    calc = GPAW(**parameters)

    GaAspris.calc = calc
    GaAspris.get_potential_energy()

    calc.write('GaAs{0}{0}{0}.pristine.gpw'.format(N))

    pristine = 'GaAs{0}{0}{0}.pristine.gpw'.format(N)
    charged = 'GaAs{0}{0}{0}.Ga_vac_charged.gpw'.format(N)
    elc = ElectrostaticCorrections(pristine=pristine,
                                   charged=charged,
                                   q=q,
                                   sigma=sigma,
                                   dimensionality='3d')
    elc.set_epsilons(epsilon)
    E_corr = elc.calculate_corrected_formation_energy()
    E_uncorr = elc.calculate_uncorrected_formation_energy()

    assert E_corr == pytest.approx(E_corr_t, abs=1e-3)
    assert E_uncorr == pytest.approx(E_uncorr_t, abs=1e-3)


if __name__ == "__main__":
    test_fnv()
