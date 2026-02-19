import pytest
from ase.build import molecule

from gpaw.new.ase_interface import GPAW


@pytest.mark.parametrize('mode', ['pw', 'fd'])
@pytest.mark.parametrize('eigensolver', ['davidson', 'rmm-diis'])
@pytest.mark.parametrize('max_mem', [-50, 0, 1024**6])
def test_max_buffer_mem(mode, eigensolver, max_mem, comm):
    atoms = molecule('H2', vacuum=1)
    domain_size = 2 if comm.size == 4 or comm.size == 8 else 1
    calc = GPAW(mode=mode,
                eigensolver={'name': eigensolver,
                             'max_buffer_mem': max_mem},
                xc='LDA',
                gpts=(12, 12, 12) if mode == 'fd' else None,
                convergence={'maximum iterations': 2},
                parallel={'domain': domain_size},
                communicator=comm,
                txt=None)
    atoms.calc = calc
    e = atoms.get_potential_energy()

    expected_e = {'pw-rmm-diis': -14.400,
                  'fd-rmm-diis': 5.9194033,
                  'pw-davidson': -16.0133410,
                  'fd-davidson': 4.6795767}
    assert e == pytest.approx(expected_e[f'{mode}-{eigensolver}'], abs=1e-3)


def test_empty_domain_rank(comm):
    # This test is mostly for MPI.
    atoms = molecule('H')
    atoms.center(vacuum=2)
    eigensolver = 'ppcg'
    calc = GPAW(mode={'name': 'pw',
                      'ecut': 1},
                eigensolver=eigensolver,
                h=0.2,
                xc='LDA',
                convergence={'maximum iterations': 2},
                parallel={'domain': comm.size},
                communicator=comm)
    atoms.calc = calc
    atoms.get_potential_energy()


if __name__ == '__main__':
    test_max_buffer_mem('fd', 'rmm-diis', -50)
