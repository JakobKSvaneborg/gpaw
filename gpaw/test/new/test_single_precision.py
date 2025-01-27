import pytest
import numpy as np

from ase.build import bulk
from ase.build import molecule

from gpaw.new.ase_interface import GPAW
from gpaw.new.pw.builder import PWDFTComponentsBuilder

def test_single_precision():
    
    
    #atoms = bulk('Ag')
    atoms = molecule('H2')
    atoms.center(vacuum=2.5)
    
    atoms.calc = GPAW(xc='PPLDA',
                      symmetry='off',
                      random=True,
                      #mode={'name': 'pw', 'ecut': 200.0})
                     # h=0.21,
                      mode={'name': 'pw', 'ecut': 200.0, 'dtype': np.float32})
    atoms.get_potential_energy()
    