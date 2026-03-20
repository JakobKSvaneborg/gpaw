import numpy as np
import pytest
from ase import Atoms

from gpaw import FermiDirac
from gpaw.mpi import world
from gpaw.response.bse import BSE
from gpaw.test import findpeak
from gpaw.utilities import compiled_with_sl
import matplotlib.pyplot as plt
from gpaw.new.symmetry import create_symmetries_object

pytestmark = pytest.mark.skipif(
    world.size < 4 or not compiled_with_sl(),
    reason='world.size < 4 or not compiled_with_sl()')


@pytest.mark.response
def test_response_bse_magnon(in_tmp_dir, mpi):
    calc = mpi.GPAW(mode='pw',
                    xc='PBE',
                    nbands='nao',
                    occupations=FermiDirac(0.001),
                    convergence={'bands': -5},
                    kpts={'size': (3, 3, 1), 'gamma': True})

    a = 3.945
    c = 8.0
    layer = Atoms(symbols='ScSe2',
                  cell=[a, a, c, 90, 90, 120],
                  pbc=(1, 1, 0),
                  scaled_positions=[(0, 0, 0),
                                    (2 / 3, 1 / 3, 0.0),
                                    (2 / 3, 1 / 3, 0.0)])
    layer.positions[1, 2] += 1.466
    layer.positions[2, 2] -= 1.466
    layer.center(axis=2)
    layer.set_initial_magnetic_moments([1.0, 0, 0])

    sym = create_symmetries_object(layer)
    print(f'ScSe2 has inversion symmetry: {sym.has_inversion}')
    print(f'Number of symmetry operations: {len(sym)}')

    layer.calc = calc
    layer.get_potential_energy()
    calc.write('ScSe2.gpw', mode='all')

    eshift = 4.2
    bse = BSE('ScSe2.gpw',
              ecut=10,
              valence_bands=[22],
              conduction_bands=[23],
              eshift=eshift,
              nbands=15,
              mode='BSE',
              truncation='2D',
              comm=mpi.comm)

    w_w = np.linspace(-2, 2, 4001)
    chi_Gw = bse.get_magnetic_susceptibility(eta=0.1,
                                             write_eig='chi+-_0_',
                                             w_w=w_w)

    w, I = findpeak(w_w, -chi_Gw[0].imag)

    fig, ax = plt.subplots()
    ax.plot(w_w, -chi_Gw[0].imag)
    ax.axvline(w, color='r', linestyle='--', label=f'peak: w={w:.4f}, I={I:.3f}')
    ax.set_xlabel('Frequency (eV)')
    ax.set_ylabel(r'$-\mathrm{Im}\,\chi^{+-}_{G=0}$')
    ax.set_title('BSE magnon spectrum (q=0)')
    ax.legend()
    fig.savefig('bse_magnon_q0.png', dpi=150)
    plt.close(fig)

    assert np.abs(w + 0.0195) < 0.001
    assert np.abs(I - 4.676) < 0.01

    bse = BSE('ScSe2.gpw',
              ecut=10,
              q_c=[1 / 3, 1 / 3, 0],
              valence_bands=[22],
              conduction_bands=[23],
              eshift=eshift,
              nbands=15,
              mode='BSE',
              truncation='2D',
              comm=mpi.comm)

    w_w = np.linspace(-2, 2, 4001)
    chi_Gw = bse.get_magnetic_susceptibility(eta=0.1,
                                             write_eig='chi+-_1_',
                                             w_w=w_w)

    w, I = findpeak(w_w, -chi_Gw[0].imag)

    fig, ax = plt.subplots()
    ax.plot(w_w, -chi_Gw[0].imag)
    ax.axvline(w, color='r', linestyle='--', label=f'peak: w={w:.4f}, I={I:.3f}')
    ax.set_xlabel('Frequency (eV)')
    ax.set_ylabel(r'$-\mathrm{Im}\,\chi^{+-}_{G=0}$')
    ax.set_title('BSE magnon spectrum (q=[1/3, 1/3, 0])')
    ax.legend()
    fig.savefig('bse_magnon_q1.png', dpi=150)
    plt.close(fig)

    assert np.abs(w + 0.0153) < 0.001
    assert np.abs(I - 7.624) < 0.01
