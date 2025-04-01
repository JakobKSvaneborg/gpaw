import numpy as np

from ase.build import bulk

from gpaw.new.builder import get_calculation_info


def test_dft_params():
    atoms = bulk('Si')
    dft_params = get_calculation_info(atoms,
                                      h=0.15,
                                      xc='PBE',
                                      kpts={'density': 2, 'gamma': True},
                                      mode='lcao',
                                      spinpol=True,)
    assert len(dft_params.ibz) == 10
    assert (dft_params.grid.size == np.array([20, 20, 20])).all()
    assert dft_params.nspins == 2
    assert dft_params.nbands == 8
