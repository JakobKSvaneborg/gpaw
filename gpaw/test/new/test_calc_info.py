import numpy as np

from ase.build import bulk

from gpaw.new.builder import get_calculation_info


def test_calc_info():
    atoms = bulk('Si')
    calc_info = get_calculation_info(atoms,
                                     h=0.15,
                                     xc='PBE',
                                     kpts={'density': 2, 'gamma': True},
                                     mode='lcao',
                                     spinpol=True,)
    assert len(calc_info.ibz) == 10
    assert (calc_info.grid.size == np.array([20, 20, 20])).all()
    assert calc_info.nspins == 2
    assert calc_info.nbands == 8
