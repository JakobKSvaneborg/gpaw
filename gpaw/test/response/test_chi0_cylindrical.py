import pytest
from ase import Atoms
from ase.units import Hartree
from ase.lattice.hexagonal import Hexagonal
from gpaw import GPAW, FermiDirac
from gpaw.response.df import DielectricFunction
from gpaw.response.qpd import SingleCylQPWDescriptor


@pytest.fixture
def gpwfile(in_tmp_dir):
    calc = GPAW(
        mode='pw',
        xc='PBE',
        nbands=16,
        convergence={'bands': 15},
        setups={'Mo': '6'},
        occupations=FermiDirac(0.001),
        kpts={'size': (3, 3, 1), 'gamma': True})

    a = 3.1604
    c = 10.0

    cell = Hexagonal(symbol='Mo',
                     latticeconstant={'a': a, 'c': c}).get_cell()
    layer = Atoms(symbols='MoS2', cell=cell, pbc=[True, True, False],
                  scaled_positions=[(0, 0, 0.5),
                                    (2 / 3, 1 / 3, 0.3 + 0.5),
                                    (2 / 3, 1 / 3, -0.3 + 0.5)])

    pos = layer.get_positions()
    pos[1][2] = pos[0][2] + 3.172 / 2
    pos[2][2] = pos[0][2] - 3.172 / 2
    layer.set_positions(pos)
    layer.calc = calc
    layer.get_potential_energy()
    fname = 'MoS2.gpw'
    calc.write(fname, mode='all')
    return fname


@pytest.mark.response
@pytest.mark.slow
def test_response_gw_MoS2_cut(gpwfile):
    ecut_sphere = 50.0

    DFs = DielectricFunction(calc=gpwfile,
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.5},
                             ecut=ecut_sphere,
                             truncation='2D',
                             hilbert=False)
    dfs1, dfs2 = DFs.get_dielectric_function()

    ecut_cyl = {
        'class': SingleCylQPWDescriptor,
        'kwargs': {'ecut_xy': ecut_sphere / Hartree,
                   'ecut_z': 0.5 * ecut_sphere / Hartree}
    }

    DFc = DielectricFunction(calc=gpwfile,
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.5},
                             ecut=ecut_cyl,
                             truncation='2D',
                             hilbert=False)
    dfc1, dfc2 = DFc.get_dielectric_function()

    assert dfc1 == pytest.approx(dfs1, rel=1e-6)
    assert dfc2 == pytest.approx(dfs2, rel=5e-2)
