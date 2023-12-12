from ase import Atoms
import numpy as np
import pytest

from gpaw.mpi import serial_comm, world
from gpaw.new.ase_interface import GPAW
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.nlopt.shg import get_shg


@pytest.mark.skipif(world.size > 4, reason='System too small')
def test_shg(in_tmp_dir):
    # Check for Hydrogen atom
    atoms = Atoms('H', cell=(3 * np.eye(3)), pbc=True)

    # Do a GS and save it
    calc = GPAW(mode={'name': 'pw', 'ecut': 600},
                communicator=serial_comm,
                symmetry={'point_group': False},
                kpts={'size': (2, 2, 2)},
                nbands=5,
                txt=None)
    atoms.calc = calc
    atoms.get_potential_energy()

    # Get the mml
    nlodata = make_nlodata(calc, comm=world)

    # Do a SHG
    freqs = np.linspace(0, 5, 101)
    get_shg(nlodata, freqs=freqs)
    get_shg(nlodata, freqs=freqs, gauge='vg', out_name='shg2.npy')

    # Check it
    if world.rank == 0:
        shg = np.load('shg.npy')
        shg2 = np.load('shg2.npy')
        # Check for nan's
        assert not np.isnan(shg).any()
        assert not np.isnan(shg2).any()
        # Check the two gauges
        assert np.all(np.abs(shg2[1] - shg[1]) < 1e-3)
        # It should be zero (small) since H2 is centro-symm.
        assert np.all(np.abs(shg[1]) < 1e-8)


def test_shg_spinpol(gpw_files, in_tmp_dir):
    shg_values = np.array([-0.77053399 - 0.37041593j,
                           -0.87903174 - 0.4177294j,
                           -1.00791251 - 0.51051291j,
                           -1.15465962 - 0.66642326j,
                           -1.30812094 - 0.92114822j,
                           -1.42138133 - 1.33424513j,
                           -1.34649601 - 1.96084827j,
                           -0.78891819 - 2.66240386j,
                           0.27801137 - 2.8572836j,
                           1.12315952 - 2.30446868j,
                           1.38569995 - 1.59698796j])

    freqs = np.linspace(2.3, 2.4, 11)
    shg_xyz = {}
    for spinpol in ['spinpaired', 'spinpol']:
        tag = '_spinpol' if spinpol == 'spinpol' else ''

        # Get nlodata from pre-calculated SiC fixtures
        nlodata = make_nlodata(gpw_files[f'sic_pw{tag}'],
                               ni=0, nf=8, comm=world)

        # Calculate 'xyz' tensor element of SHG spectra
        get_shg(nlodata, freqs=freqs, eta=0.025, pol='xyz',
                out_name=f'shg_xyz{tag}.npy')
        world.barrier()

        # Load the calculated SHG spectra (in units of nm/V)
        shg_xyz[spinpol] = np.load(f'shg_xyz{tag}.npy')[1] * 1e9
        assert shg_xyz[spinpol] == pytest.approx(shg_values, abs=1e-3), \
            np.max(np.abs(shg_xyz[spinpol] - shg_values))

    # import matplotlib.pyplot as plt
    # plt.plot(freqs, shg_xyz['spinpaired'])
    # plt.plot(freqs, shg_xyz['spinpol'])
    # plt.show()

    # Assert that the difference between spectra from spinpaired and
    # spinpolarised calculations is small

    # Absolute error
    shg_xyz_diff = shg_xyz['spinpaired'] - shg_xyz['spinpol']
    assert shg_xyz_diff.real == pytest.approx(0, abs=5e-4)
    assert shg_xyz_diff.imag == pytest.approx(0, abs=5e-4)

    # Relative error
    shg_xyz_avg = (shg_xyz['spinpaired'] + shg_xyz['spinpol']) / 2
    shg_xyz_rerr_real = shg_xyz_diff.real / shg_xyz_avg.real
    shg_xyz_rerr_imag = shg_xyz_diff.imag / shg_xyz_avg.imag
    assert shg_xyz_rerr_real == pytest.approx(0, abs=2e-3), \
        np.max(np.abs(shg_xyz_rerr_real))
    assert shg_xyz_rerr_imag == pytest.approx(0, abs=2e-3), \
        np.max(np.abs(shg_xyz_rerr_imag))
