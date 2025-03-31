from ase.build import bulk

from gpaw.new.ase_interface import GPAW


def test_eig_convergence():
    atoms = bulk('Si')
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                      convergence={'eigenvalues': 1e-3},
                      )
    atoms.get_potential_energy()

    atoms = bulk('Si')
    atoms.calc = GPAW(mode={'name': 'lcao'},
                      basis='sz(dzp)',
                      convergence={'eigenvalues': 1e-3},
                      )
    atoms.get_potential_energy()
