import numpy as np
import pytest
from ase import Atoms

from gpaw import GPAW, Mixer
from gpaw.cdft.cdft import CDFT


@pytest.mark.old_gpaw_only
def test_cdft_fd_forces(in_tmp_dir):

    sys = Atoms('N2', positions=([0., 0., 0.], [0., 0., 1]))
    sys.center(3)
    sys.set_pbc(False)
    sys.set_initial_magnetic_moments([0.5, 0.5])

    calc_b = GPAW(legacy_gpaw=True,
                  h=0.2,
                  basis='dzp',
                  charge=1,
                  mode='lcao',
                  xc='PBE',
                  spinpol=True,
                  mixer=Mixer(beta=0.25, nmaxold=3, weight=100.0),
                  txt='N2.txt',
                  convergence={'eigenstates': 1.0e-2, 'density': 1.0e-2,
                               'energy': 1e-2})

    sys.calc = calc_b
    sys.get_potential_energy()

    cdft_b = CDFT(calc=calc_b,
                  atoms=sys,
                  charge_regions=[[0]],
                  charges=[1],
                  charge_coefs=[22],
                  forces='fd',
                  method='L-BFGS-B',
                  txt='N2.cdft',
                  minimizer_options={'gtol': 0.1})

    sys.calc = cdft_b
    sys.get_potential_energy()
    f = sys.get_forces()
    # check that forces in x and y directions are small
    for i in [0, 1]:
        assert (np.isclose(f[0, i], 0.))
        assert (np.isclose(f[1, i], 0.))
