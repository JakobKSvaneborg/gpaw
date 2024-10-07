import pytest
from ase import Atoms
import numpy as np
from gpaw import GPAW, FermiDirac, Davidson, Mixer, restart
from gpaw.cdft.cdft import CDFT
from gpaw.cdft.cdft_coupling import CouplingParameters
from gpaw.mpi import world


@pytest.mark.old_gpaw_only
@pytest.mark.skipif(world.size > 1, reason='cdft coupling not parallel')
def test_cdft_restart(in_tmp_dir):
    distance = 2.5
    sys = Atoms('He2', positions=([0., 0., 0.], [0., 0., distance]))
    sys.center(3)
    sys.set_pbc(False)
    sys.set_initial_magnetic_moments([0.5, 0.5])

    calc_b = GPAW(h=0.2,
                  mode='fd',
                  basis='dzp',
                  charge=1,
                  xc='PBE', symmetry='off',
                  occupations=FermiDirac(0., fixmagmom=True),
                  eigensolver=Davidson(3),
                  spinpol=True,
                  nbands=4,
                  mixer=Mixer(beta=0.25, nmaxold=3, weight=100.0),
                  txt='He2+_final_%3.2f.txt' % distance,
                  convergence={'eigenstates': 1.0,
                               'density': 1.0,
                               'energy': 1.0,
                               'bands': 4})

    cdft_b = CDFT(calc=calc_b,
                  atoms=sys,
                  charge_regions=[[1], [0]],
                  charges=[1, 0],
                  charge_coefs=[27, 0],
                  method='L-BFGS-B',
                  txt='He2+_final_%3.2f.cdft' % distance,
                  minimizer_options={'gtol': 0.1})
    sys.calc = cdft_b
    sys.get_potential_energy()
    sys.calc.calc.write('He2.gpw', mode='all')

    # Restart

    atoms, calc = restart('He2.gpw')

    coupling = CouplingParameters(calc_a=calc, calc_b=calc,
                                  wfs_a='He2.gpw', wfs_b='He2.gpw',
                                  Va=[27, 0], Vb=[27, 0],
                                  charge_regions_A=[[1], [0]],
                                  charge_regions_B=[[1], [0]])
    overlaps = coupling.get_pair_density_matrix(calc, calc)[0]
    for i in [0, 1, 2]:
        assert (np.isclose(np.real(overlaps[0, i, i]), 1.))
