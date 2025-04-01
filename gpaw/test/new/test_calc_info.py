import numpy as np

from ase.build import bulk

from gpaw.new.builder import get_calculation_info


def test_calc_info():
    atoms = bulk('Si')
    info = get_calculation_info(atoms,
                                h=0.3,
                                xc='LDA',
                                kpts={'density': 1, 'gamma': True},
                                mode={'name': 'lcao'},
                                basis='sz(dzp)',
                                spinpol=True,)
    assert len(info.ibz) == 4
    assert (info.grid.size == np.array([12, 12, 12])).all()
    assert info.nspins == 2
    assert info.nbands == 8

    assert info.get_dft_calc() is not None
    calc = info.get_ase_calc({'parallel': {'band': 1}})
    calc.get_potential_energy()

    info2 = info.update_params(mode={'name': 'pw'})
    info2.get_ase_calc().get_potential_energy()
