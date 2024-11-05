import pytest
from ase.build import bulk
from gpaw.new.ase_interface import GPAW


@pytest.fixture(scope='module')
def noprojs_gpw(module_tmp_path):
    atoms = bulk('Si')
    atoms.calc = GPAW(mode='pw', kpts=[2, 2, 2], txt=None)
    atoms.get_potential_energy()
    gpw_path = module_tmp_path / 'gs_noprojs.gpw'
    atoms.calc.write(gpw_path, include_projections=False)
    return gpw_path


def test_no_save_projections(noprojs_gpw):
    calc = GPAW('gs_noprojs.gpw')
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


# Test that we can do fixed_density()
# Also: test lcao, fd, pw
