import numpy as np

from ase.build import bulk

from gpaw.new.builder import get_dft_parameters

def test_dft_params():
    atoms = bulk('Si')
    dft_params = get_dft_parameters(atoms,
                                    xc='PBE',
                                    kpts={'density': 2, 'gamma': True},
                                    mode='pw')
    assert len(dft_params.ibz) == 10
    assert (dft_params.grid.size == np.array([18, 18, 18])).all()
    breakpoint()
    