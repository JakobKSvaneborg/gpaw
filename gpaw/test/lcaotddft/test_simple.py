import numpy as np
import pytest
from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world
from gpaw.poisson import PoissonSolver
from gpaw.tddft.spectrum import photoabsorption_spectrum


@pytest.mark.rttddft
def test_lcaotddft_simple(in_tmp_dir):
    atoms = molecule('Na2')
    atoms.center(vacuum=4.0)

    # Ground-state calculation
    calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
                basis='dzp', mode='lcao',
                poissonsolver=PoissonSolver('fd', eps=1e-16),
                convergence={'density': 1e-8},
                txt='gs.out',
                symmetry={'point_group': False})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    # Time-propagation calculation
    td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 3)
    photoabsorption_spectrum('dm.dat', 'spec.dat', delta_e=5)
    world.barrier()

    # Test dipole moment
    data_i = np.loadtxt('dm.dat')[:, 2:].ravel()
    if 0:
        from gpaw.test import print_reference
        print_reference(data_i, 'ref_i', '%.12le')

    ref_i = [-9.383700894739e-16,
             -9.338586948130e-16,
             2.131582675483e-14,
             8.679923327633e-15,
             7.529517689096e-15,
             2.074867751820e-14,
             1.967175558125e-05,
             1.967175557952e-05,
             1.805004256446e-05,
             3.799528978877e-05,
             3.799528978943e-05,
             3.602506734201e-05,
             5.371491974467e-05,
             5.371491974534e-05,
             5.385046706407e-05]

    tol = 1e-10
    assert data_i == pytest.approx(ref_i, abs=tol)

    # Test spectrum
    data_i = np.loadtxt('spec.dat').ravel()
    if 0:
        from gpaw.test import print_reference
        print_reference(data_i, 'ref_i', '%.12le')

    ref_i = [0.000000000000e+00,
             0.000000000000e+00,
             0.000000000000e+00,
             0.000000000000e+00,
             5.000000000000e+00,
             4.500226856200e-03,
             4.500226856200e-03,
             4.408379542600e-03,
             1.000000000000e+01,
             1.659426124300e-02,
             1.659426124300e-02,
             1.623812256800e-02,
             1.500000000000e+01,
             3.244686838800e-02,
             3.244686838800e-02,
             3.168682490900e-02,
             2.000000000000e+01,
             4.684883744600e-02,
             4.684883744600e-02,
             4.559689861200e-02,
             2.500000000000e+01,
             5.466781222200e-02,
             5.466781222200e-02,
             5.290171209500e-02,
             3.000000000000e+01,
             5.231586230700e-02,
             5.231586230700e-02,
             5.008661764300e-02]

    tol = 1e-7
    assert data_i == pytest.approx(ref_i, abs=tol)


@pytest.mark.rttddft
def test_lcaotddft_fail_with_symmetry(in_tmp_dir):
    atoms = molecule('Na2')
    atoms.center(vacuum=4.0)

    # Ground-state calculation with point group symmetries
    calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
                basis='dzp', mode='lcao',
                poissonsolver=PoissonSolver('fd', eps=1e-16),
                convergence={'density': 1e-8},
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    # Time-propagation calculation
    # should not be allowed with symmetries
    with pytest.raises(ValueError):
        LCAOTDDFT('gs.gpw', txt='td.out')
