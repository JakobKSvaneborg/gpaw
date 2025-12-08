import pytest
from ase.build import molecule

from gpaw import Mixer
from gpaw.utilities.elpa import LibElpa

pytestmark = pytest.mark.skipif(not LibElpa.have_elpa(),
                                reason='not LibElpa.have_elpa()')


def test_lcao_lcao_elpa(mpi, require_real_mpi):
    size = (mpi.comm.size // 2, 2) if mpi.comm.size > 1 else (1, 1)

    energies = []
    for use_elpa in [1, 0]:
        atoms = molecule('CH3CH2OH', vacuum=2.5)
        calc = mpi.GPAW(
            mode='lcao', basis='dzp',
            h=0.25,
            parallel=dict(sl_default=(size[0], size[1], 3),
                          use_elpa=use_elpa),
            mixer=Mixer(0.5, 5, 50.0))
        atoms.calc = calc
        E = atoms.get_potential_energy()
        energies.append(E)
    err = abs(energies[1] - energies[0])
    assert err < 1e-10, err
