import numpy as np
import pytest

from ase.build import bulk

from gpaw import get_calculation_info
from gpaw.mpi import world


def test_calc_info():
    atoms = bulk('Si')
    info = get_calculation_info(atoms,
                                h=0.3,
                                xc='LDA',
                                kpts={'density': 1, 'gamma': True},
                                mode={'name': 'lcao'},
                                basis='sz(dzp)',
                                spinpol=True,
                                comm=world,
                                txt=None)
    assert len(info.ibz) == 4
    assert (info.grid.size == np.array([12, 12, 12])).all()
    assert info.nspins == 2
    assert info.nbands == 8
    assert info.wf_description is None
    assert info.dft_calculation() is not None

    atoms.calc = info.ase_calculator()
    atoms.get_potential_energy()

    info2 = info.update_params(mode={'name': 'pw'})
    assert info2.wf_description is not None
    atoms.calc = info2.ase_calculator()

    with pytest.raises(TypeError):
        get_calculation_info(atoms, atoms)
