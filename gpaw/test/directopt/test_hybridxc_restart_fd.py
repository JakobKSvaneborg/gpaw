import pytest
from ase import Atoms
from ase.units import Bohr

from gpaw import FD, GPAW
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.xc.hybrid import HybridXC


@pytest.mark.do
def test_hybridxc_restart_fd(in_tmp_dir):
    """Test that FD hybrid functional eigenvalues survive .gpw restart.

    Verifies fix for TheochemUI/gpaw#14: set_positions_without_ruining_
    everything() did not call xc.set_positions(), leaving HybridXC.ghat
    with uninitialized positions on restart. This caused wrong exact
    exchange matrix elements and ~0.6 eV eigenvalue shifts.
    """
    d = 1.4 * Bohr
    h2 = Atoms('H2',
               positions=[[-d / 2, 0, 0],
                          [d / 2, 0, 0]])
    h2.center(vacuum=3)

    calc = GPAW(mode=FD(),
                h=0.3,
                xc=HybridXC('PBE0', unocc=True),
                eigensolver=FDPWETDM(converge_unocc=True),
                mixer={'backend': 'no-mixing'},
                occupations={'name': 'fixed-uniform'},
                symmetry='off',
                nbands=3,
                convergence={'eigenstates': 4.0e-6})
    h2.calc = calc
    e1 = h2.get_potential_energy()
    eig1 = calc.get_eigenvalues()[:2]

    calc.write('h2_pbe0.gpw', mode='all')

    # Restart and verify eigenvalues match
    calc2 = GPAW('h2_pbe0.gpw')
    eig2 = calc2.get_eigenvalues()[:2]

    assert eig2 == pytest.approx(eig1, abs=1e-6), (
        f'Eigenvalue mismatch after restart: {eig2} vs {eig1}')

    # Also verify energy is accessible
    e2 = calc2.results['energy']
    assert e2 == pytest.approx(e1, abs=1e-6)
