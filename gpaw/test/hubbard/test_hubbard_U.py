from math import sqrt

from ase import Atoms
from ase.dft.bandgap import bandgap

from gpaw import GPAW, FermiDirac
import pytest


def test_Hubbard_U(in_tmp_dir):
    """Setup up bulk NiO in an antiferromagnetic configuration."""
    # Lattice constant:
    a = 4.19
    b = a / sqrt(2)

    m = 2
    k = 2  # number of k-points

    atoms = Atoms(
        symbols='Ni2O2',
        pbc=True,
        cell=(b, b, a),
        positions=[(0, 0, 0),
                   (b / 2, b / 2, a / 2),
                   (0, 0, a / 2),
                   (b / 2, b / 2, 0)],
        magmoms=(m, -m, 0, 0))

    name = 'ni2o2'
    for setup in ['10', '10:d,6.0']:
        calc = GPAW(
            mode='pw',
            occupations=FermiDirac(width=0.05),
            setups={'Ni': setup},
            convergence={'eigenstates': 8e-4,
                         'density': 1.0e-2,
                         'energy': 0.1},
            txt=name + '.txt',
            kpts=(k, k, k),
            xc='oldPBE')
        atoms.calc = calc
        atoms.get_potential_energy()
        gap, _, _ = bandgap(calc)
        print(name, gap)
        if name == 'ni2o2':
            assert gap == pytest.approx(0.8, abs=0.1)
        else:
            assert gap == pytest.approx(4.7, abs=0.2)
        name += '+U'
