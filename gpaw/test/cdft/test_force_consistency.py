import numpy as np
import pytest
from ase import Atoms
from ase.calculators.fd import calculate_numerical_forces
from gpaw import GPAW
from gpaw.cdft.cdft import CDFT
from gpaw.directmin.etdm_fdpw import FDPWETDM


@pytest.mark.old_gpaw_only
def test_cdft_forces_consistency(in_tmp_dir, comm):
    delta = 0.01
    charge_regions = [[0], [1]]
    charge = [0.4, -0.2]
    coefs = [0, 0]

    atoms = Atoms('COO',
                  [[3.1, 2.98, 3.12],
                   [2.92, 3.00, 4.25],
                   [2.95, 2.97, 1.83]],
                  cell=[6, 6, 6])

    calc_ground = GPAW(mode='fd',
                       h=0.2,
                       xc='PBE',
                       spinpol=True,
                       txt='ground_state_output_A.txt',
                       eigensolver=FDPWETDM(converge_unocc=False),
                       mixer={'backend': 'no-mixing'},
                       occupations={'name': 'fixed-uniform'},
                       communicator=comm)

    calc_cdft = CDFT(calc=calc_ground,
                     atoms=atoms,
                     charge_regions=charge_regions,
                     charges=charge,
                     charge_coefs=coefs,
                     method='L-BFGS-B',
                     txt='cdftA_output.txt',
                     minimizer_options={'gtol':0.001})

    atoms.calc = calc_cdft
    f_cdft = atoms.get_forces()[0, 2]
    f_finite_difference = calculate_numerical_forces(atoms, delta, [0], [2])

    assert (np.abs(f_finite_difference-f_cdft)<0.1)
