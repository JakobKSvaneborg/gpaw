import pytest
from ase.build import molecule
from gpaw.new.ase_interface import GPAW


@pytest.mark.parametrize('mode', ['pw', 'lcao', 'fd'])
@pytest.mark.parametrize('kpts', [False, True])
def test_write_new_single(mode, kpts, in_tmp_dir):
    atoms = molecule('H2')
    atoms.center(vacuum=2.0)
    if kpts:
        atoms.pbc = True
        atoms.calc = GPAW(mode=mode, kpts=(2, 1, 1), txt=None)
    else:
        atoms.calc = GPAW(mode=mode, txt=None)
    atoms.get_potential_energy()

    # write
    atoms.calc.write('h2.gpw', mode='all', precision='single')
    dft1 = atoms.calc.dft
    # load
    calc = GPAW('h2.gpw')
    dft2 = calc.dft
    assert dft1.density.nt_sR.data == pytest.approx(dft2.density.nt_sR.data)
    assert dft1.potential.vt_sR.data == pytest.approx(
        dft2.potential.vt_sR.data)
    for wfs1, wfs2 in zip(dft1.ibzwfs, dft2.ibzwfs):
        if mode == 'lcao':
            assert wfs1.C_nM.data == pytest.approx(wfs2.C_nM.data)
        else:
            assert wfs1.psit_nX.data == pytest.approx(wfs2.psit_nX.data)
