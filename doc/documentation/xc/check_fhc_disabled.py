"""check is libxc is compiled with --disable-fhc (needed for mggas)"""
from ase.build import molecule
from gpaw import GPAW, KohnShamConvergenceError
from gpaw.utilities.adjust_cell import adjust_cell

vacuum = 4.0
h = 0.3


def test_mgga_lxc_fhc():
    cluster = molecule('CO')
    adjust_cell(cluster, border=vacuum, h=h)
    calc = GPAW(xc='MGGA_X_TPSS+MGGA_C_TPSS',
                mode='fd',
                h=h,
                maxiter=14,
                convergence={
                    'energy': 0.5,
                    'density': 1.0e-1,
                    'eigenstates': 4.0e-1})
    cluster.calc = calc
    try:
        cluster.get_potential_energy()
    except KohnShamConvergenceError:
        pass
    assert calc.scf.converged


if __name__ == '__main__':
    test_mgga_lxc_fhc()
