import pytest
from ase.build import bulk
from gpaw.new.ase_interface import GPAW


@pytest.fixture(scope='module', params=['fd', 'lcao', 'pw'])
def noprojs_gpw(module_tmp_path, request):
    mode = request.param
    atoms = bulk('Si')
    atoms.calc = GPAW(mode=mode, kpts=[2, 2, 2], txt=None)
    atoms.get_potential_energy()
    gpw_path = module_tmp_path / 'gs_noprojs_{mode}.gpw'
    atoms.calc.write(gpw_path, include_projections=False)
    return gpw_path


def test_no_save_projections(noprojs_gpw):
    calc = GPAW(noprojs_gpw)
    ibzwfs = list(calc.dft.ibzwfs)
    assert len(ibzwfs) > 0
    for wfs in ibzwfs:
        assert wfs._P_ani is None


def test_nice_error_message(noprojs_gpw):
    # We want there to be a good error message when we do not have
    # projections.  This only tests the most obvious case of .P_ani access,
    # but there could be code paths that will crash less controllably.
    calc = GPAW(noprojs_gpw)

    wfs = next(iter(calc.dft.ibzwfs))
    with pytest.raises(RuntimeError, match='There are no proj'):
        wfs.P_ani


def test_fixed_density_bandstructure(tmp_path, noprojs_gpw):
    calc = GPAW(noprojs_gpw)

    fixed_calc = calc.fixed_density(
        kpts=[[0., 0., 0.], [0., 0., 0.5]], symmetry='off')

    bs = fixed_calc.band_structure()
    assert len(bs.path.kpts) == 2
    ibzwfs = list(fixed_calc.dft.ibzwfs)

    for wfs in ibzwfs:
        assert len(wfs.P_ani) == len(calc.get_atoms())
    # Should we test something else here?
    # If we calculate a full bandstructure, it looks realistic.
    # We could compare to an "ordinary" (with projections) gpw file
    # to see that the numbers are in fact unaffected by the distinction.

# Remember to test in parallel, too
