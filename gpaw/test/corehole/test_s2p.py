from ase.build import molecule
from gpaw import GPAW
from gpaw.test import gen
from gpaw.xas import XAS


def test_sulphur_1s_xas(in_tmp_dir, add_cwd_to_setup_paths):
    atoms = molecule('SH2')
    atoms.center(3)

    setupname = 'S1s1ch'
    gen('S', name=setupname, corehole=(1, 0, 1), gpernode=30, write_xml=True)

    nbands = 6
    nocc = 4  # for SH2
    atoms.calc = GPAW(mode='fd', h=0.3, nbands=nbands,
                      setups={'S': setupname}, txt=None)
    atoms.get_potential_energy()

    xas = XAS(atoms.calc)
    x, y = xas.get_spectra(stick=True, proj_xyz=True)
    print('############## x=', x)
    print('############## y=', y)
    assert y.shape == (3, 1, nbands - nocc)  # why this shape ?


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
