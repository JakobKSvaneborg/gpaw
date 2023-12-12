import numpy as np
import pytest
from ase.build import molecule
from ase.units import Bohr, Hartree
from gpaw import GPAW
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.laser import GaussianPulse
from gpaw.mpi import world
from gpaw.tddft.units import as_to_au

# Settings
dt = 20.0
N1 = 5
N = 5 + N1
kick_v = np.ones(3) * 1e-5


@pytest.mark.rttddft
def test_laser(in_tmp_dir):
    # Atoms
    atoms = molecule('Na2')
    atoms.center(vacuum=4.0)

    # Ground-state calculation
    calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
                basis='dzp', mode='lcao',
                convergence={'density': 1e-8},
                symmetry={'point_group': False},
                txt='gs.out')
    atoms.calc = calc
    _ = atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    # Time-propagation calculation
    td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick(kick_v)
    td_calc.propagate(dt, N)

    # Pulse
    direction = kick_v
    ext = ConstantElectricField(Hartree / Bohr, direction)
    pulse = GaussianPulse(1e-5, 0e3, 8.6, 0.5, 'sin')

    # Time-propagation calculation with pulse
    td_calc = LCAOTDDFT('gs.gpw', td_potential={'ext': ext, 'laser': pulse},
                        txt='tdpulse.out')
    DipoleMomentWriter(td_calc, 'dmpulse.dat')
    td_calc.propagate(dt, N1)
    td_calc.write('td.gpw', mode='all')
    # Restart
    td_calc = LCAOTDDFT('td.gpw', txt='tdpulse2.out')
    DipoleMomentWriter(td_calc, 'dmpulse.dat')
    td_calc.propagate(dt, N - N1)

    # Convoluted dipole moment
    world.barrier()
    time_t = np.arange(0, dt * (N + 0.1), dt) * as_to_au
    pulse_t = pulse.strength(time_t)
    np.savetxt('pulse.dat', np.stack((time_t, pulse_t)).T)
    dm_tv = np.delete(np.loadtxt('dm.dat')[:, 2:], 1, axis=0)
    dm_tv /= np.linalg.norm(kick_v)
    pulsedm_tv = np.delete(np.loadtxt('dmpulse.dat')[:, 2:], N1, axis=0)

    tol = 5e-6
    for v in range(3):
        pulsedmconv_t = np.convolve(
            dm_tv[:, v], pulse_t)[:(N + 1)] * dt * as_to_au
        np.savetxt('dmpulseconv%d.dat' % v, pulsedmconv_t)
        assert pulsedm_tv[:, v] == pytest.approx(pulsedmconv_t, abs=tol)
