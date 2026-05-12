import pytest
from ase.build import molecule

from gpaw import FD


def test_complex(in_tmp_dir, mpi):
    Eini0 = -17.8037610364
    energy_eps = 0.0005

    calc = mpi.GPAW(
        xc='LDA',
        h=0.21,
        convergence={'eigenstates': 3.5e-5, 'energy': energy_eps},
        mode=FD(force_complex_dtype=True))

    mol = molecule('N2')
    mol.center(vacuum=3.0)
    mol.calc = calc

    Eini = mol.get_potential_energy()
    assert Eini == pytest.approx(
        Eini0, abs=energy_eps * calc.get_number_of_electrons())

    calc.write('N2_complex.gpw', mode='all')

    mol, calc = mpi.restart('N2_complex.gpw')

    calc.dft.converge({'eigenstates': 3.5e-9,
                       'energy': energy_eps})
    assert calc.dft.ibzwfs.dtype == complex
    E = mol.get_potential_energy()
    assert E == pytest.approx(
        Eini, abs=energy_eps * calc.get_number_of_electrons())
