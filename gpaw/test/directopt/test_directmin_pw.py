import pytest
import numpy as np

from gpaw import GPAW, PW
from ase.build import molecule
from gpaw.mpi import world


@pytest.mark.new_gpaw_ready
@pytest.mark.do
@pytest.mark.parametrize('mode', ['pw'])
def test_directmin_pw(in_tmp_dir, mode, gpaw_new):
    if gpaw_new and (world.size > 1 or mode != 'pw'):
        pytest.skip('Does not work yet for new GPAW')

    atoms = molecule('H2')
    atoms.center(vacuum=4.0)
    atoms.set_pbc(False)

    if mode == 'pw':
        kwargs = dict(mode=PW(300, force_complex_dtype=True))
        e0 = -6.586933
        f0 = np.array([[0., 0., 0.61711],
                       [0., 0., -0.61711]])
    else:
        pass

    calc = GPAW(**kwargs,
                xc='PBE',
                occupations={'name': 'fixed-uniform'},
                eigensolver={'name': 'etdm-fdpw',
                             'converge_unocc': True},  # not gpaw_new},
                mixer={'backend': 'no-mixing'},
                spinpol=True,
                symmetry='off',
                nbands=-3,
                convergence={'eigenstates': 4.0e-6})
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    f = atoms.get_forces()

    assert energy == pytest.approx(e0, abs=1.0e-4)
    assert f0 == pytest.approx(f, abs=1e-2)

    if gpaw_new:
        # restart fails because of missing 'converge_unocc'
        return

    calc.write('H2.gpw', mode='all')
    from gpaw import restart
    atoms, calc = restart('H2.gpw', txt='-')
    atoms.positions += 1.0e-6
    f2 = atoms.get_forces()
    niter = calc.get_number_of_iterations()

    assert niter == pytest.approx(3, abs=1)
    assert f0 == pytest.approx(f2, abs=1e-2)


if __name__ == '__main__':
    test_directmin_pw(1, 'pw', 1)
