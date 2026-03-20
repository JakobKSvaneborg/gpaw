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
    w_w = np.linspace(-2, 2, 4001)

    # --- q = 0 ---
    bse = BSE('ScSe2.gpw',
              ecut=10,
              valence_bands=[22],
              conduction_bands=[23],
              eshift=eshift,
              nbands=15,
              mode='BSE',
              truncation='2D',
              comm=mpi.comm)

    chi_Gw = bse.get_magnetic_susceptibility(eta=0.1,
                                             write_eig='chi+-_0_',
                                             w_w=w_w)
    spectrum_fixed_q0 = -chi_Gw[0].imag

    # Re-run with sign=+1 (pre-fix behaviour)
    bse = BSE('ScSe2.gpw',
              ecut=10,
              valence_bands=[22],
              conduction_bands=[23],
              eshift=eshift,
              nbands=15,
              mode='BSE',
              truncation='2D',
              comm=mpi.comm)
    _orig = bse.get_density_matrix

    def _patched_q0(*args, **kwargs):
        rho, iq, _sign = _orig(*args, **kwargs)
        return rho, iq, 1  # force sign=+1

    bse.get_density_matrix = _patched_q0
    chi_Gw_old = bse.get_magnetic_susceptibility(eta=0.1,
                                                 w_w=w_w)
    spectrum_old_q0 = -chi_Gw_old[0].imag

    w_fix, I_fix = findpeak(w_w, spectrum_fixed_q0)
    w_old, I_old = findpeak(w_w, spectrum_old_q0)

    fig, ax = plt.subplots()
    ax.plot(w_w, spectrum_fixed_q0,
            label=f'with fix (peak: w={w_fix:.4f}, I={I_fix:.3f})')
    ax.plot(w_w, spectrum_old_q0, '--',
            label=f'without fix (peak: w={w_old:.4f}, I={I_old:.3f})')
    ax.set_xlabel('Frequency (eV)')
    ax.set_ylabel(r'$-\mathrm{Im}\,\chi^{+-}_{G=0}$')
    ax.set_title('BSE magnon spectrum (q=0)')
    ax.legend()
    fig.savefig('bse_magnon_q0.png', dpi=150)
    plt.close(fig)

    assert np.abs(w_fix + 0.0195) < 0.001
    assert np.abs(I_fix - 4.676) < 0.01

    # --- q = [1/3, 1/3, 0] ---
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

    chi_Gw = bse.get_magnetic_susceptibility(eta=0.1,
                                             write_eig='chi+-_1_',
                                             w_w=w_w)
    spectrum_fixed_q1 = -chi_Gw[0].imag

    # Re-run with sign=+1 (pre-fix behaviour)
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
    _orig = bse.get_density_matrix

    def _patched_q1(*args, **kwargs):
        rho, iq, _sign = _orig(*args, **kwargs)
        return rho, iq, 1  # force sign=+1

    bse.get_density_matrix = _patched_q1
    chi_Gw_old = bse.get_magnetic_susceptibility(eta=0.1,
                                                 w_w=w_w)
    spectrum_old_q1 = -chi_Gw_old[0].imag

    w_fix, I_fix = findpeak(w_w, spectrum_fixed_q1)
    w_old, I_old = findpeak(w_w, spectrum_old_q1)

    fig, ax = plt.subplots()
    ax.plot(w_w, spectrum_fixed_q1,
            label=f'with fix (peak: w={w_fix:.4f}, I={I_fix:.3f})')
    ax.plot(w_w, spectrum_old_q1, '--',
            label=f'without fix (peak: w={w_old:.4f}, I={I_old:.3f})')
    ax.set_xlabel('Frequency (eV)')
    ax.set_ylabel(r'$-\mathrm{Im}\,\chi^{+-}_{G=0}$')
    ax.set_title('BSE magnon spectrum (q=[1/3, 1/3, 0])')
    ax.legend()
    fig.savefig('bse_magnon_q1.png', dpi=150)
    plt.close(fig)

    assert np.abs(w_fix + 0.0153) < 0.001
    assert np.abs(I_fix - 7.624) < 0.01
