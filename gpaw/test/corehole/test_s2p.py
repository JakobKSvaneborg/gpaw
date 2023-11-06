import pytest
from ase.build import molecule
from gpaw import GPAW
from gpaw.test import gen
from gpaw.xas import XAS


@pytest.mark.skip(reason='redo the pre-calculated values')
def test_sulphur_1s_xas(in_tmp_dir, add_cwd_to_setup_paths):
    atoms = molecule('SH2')
    atoms.center(3)

    setupname = 'S1s1ch'
    gen('S', name=setupname, corehole=(1, 0, 1), gpernode=30, write_xml=True)

    atoms.calc = GPAW(mode='fd', h=0.3, setups={'S': setupname}, txt=None)
    atoms.get_potential_energy()

    xas = XAS(atoms.calc)
    x, y = xas.get_spectra(stick=True, proj_xyz=True)
    print('############## x=', x)
    print('############## y=', y)

    # pre-calculated values, only two contributions due to symmetry
    #          direction value
    pre_calc = [[1, 1.14321522e-04],
                [2, 7.88607898e-05]]
    for i, peak in enumerate(pre_calc):
        assert y[peak[0], i] == pytest.approx(peak[1])
        y[peak[0], i] = 0
    assert y == pytest.approx(0, abs=1e-20)


def test_sulphur_2p_xas(in_tmp_dir, add_cwd_to_setup_paths):
    atoms = molecule('SH2')
    atoms.center(3)

    setupname = 'S2p1ch'
    gen('S', name=setupname, corehole=(2, 1, 1), gpernode=30, write_xml=True)

    atoms.calc = GPAW(mode='fd', h=0.3, setups={'S': setupname}, txt=None)
    atoms.get_potential_energy()

    xas = XAS(atoms.calc)
    x, y = xas.get_spectra()

    # TODO we need some assert here to test validity
