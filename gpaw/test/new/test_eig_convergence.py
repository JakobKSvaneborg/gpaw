import numpy as np

from ase.build import bulk

from gpaw.new.ase_interface import GPAW
from gpaw.convergence_criteria import Criterion

def test_eig_convergence():
    class EigConvergence(Criterion):
        name = 'Eigenvalue convergence'
        tablename = 'eigs'
        calc_last = False
        
        def __init__(self, tol = 1e-8):
            self.tol = tol
            self.description = 'Eigenvalue convergence criterion'
        
        def __call__(self, context):
            eig_error = context.eigs
            converged = (eig_error < self.tol)
            entry = f'{np.log10(eig_error):+5.2f}'
            return converged, entry

    eig_conv = EigConvergence()
    
    atoms = bulk('Si')
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                      convergence={'custom': [eig_conv]},
                      )
    atoms.get_potential_energy()
    
    atoms = bulk('Si')
    atoms.calc = GPAW(mode={'name': 'lcao'},
                      basis='sz(dzp)',
                      convergence={'custom': [eig_conv]},
                      )
    atoms.get_potential_energy()