import pytest
from ase.build import molecule
from gpaw import GPAW
from gpaw.test import gen
from gpaw.xas import XAS


def folding_is_normalized(xas: XAS, rel: float = 1e-5) -> bool:
    xs, ys_cmn = xas.stick()
    ys_summed_c = ys_cmn.sum(axis=1).sum(axis=1)
    xf, yf_cn = xas.get_spectra(fwhm=0.5)
    dxf = xf[1:] - xf[:-1]
    assert dxf == pytest.approx(dxf[0])
    yf_summed_c = yf_cn.sum(axis=1) * dxf[0]

    return yf_summed_c == pytest.approx(ys_summed_c, rel=rel)


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
    x, y_cmn = xas.stick(proj_xyz=True)
    assert y_cmn.shape == (3, 1, nbands - nocc)

    assert folding_is_normalized(xas)


def test_sulphur_2p_xas(in_tmp_dir, add_cwd_to_setup_paths):
    atoms = molecule('SH2')
    atoms.center(3)

    setupname = 'S2p1ch'
    gen('S', name=setupname, corehole=(2, 1, 1), gpernode=30, write_xml=True)

    atoms.calc = GPAW(mode='fd', h=0.3, setups={'S': setupname}, txt=None)
    atoms.get_potential_energy()

    xas = XAS(atoms.calc)
    assert folding_is_normalized(xas)
