# %%
"""
# Quasiparticle bandgap using GW approximation
"""

# %%
"""
In this exercise we will calculate the quasiparticle band gap of the compound using the GW approximation. For a brief introduction to the GW theory and the details of its implementation in GPAW, see https://gpaw.readthedocs.io/documentation/gw_theory/gw_theory.html

First, we need to do a regular groundstate calculation. We do this in plane wave mode and choose the LDA exchange-correlation functional. In order to keep the computational efforts small, you should start with reasonable k-points and plane wave basis.
"""

from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw import PW

atoms = bulk('Si', 'diamond', a=5.4) # student:

calc = GPAW(mode=PW(300),  # student: # energy cutoff for plane wave basis (in eV)
            kpts={'size': (3, 3, 3), 'gamma': True}, # student: kpts={'size': (?, ?, ?), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.001),
            parallel={'domain': 1},
            txt='Si_groundstate.txt') # student: txt='???_groundstate.txt'

atoms.calc = calc
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()  # determine all bands
calc.write('Si_groundstate.gpw', 'all')  # student: '???_groundstate.gpw', 'all' # write out wavefunctions 

# %%
"""
Next, we set up the G0W0 calculator and calculate the quasi-particle spectrum for all the k-points present in the irreducible Brillouin zone from the ground state calculation and the specified bands. For example Carbon has 4 valence electrons and the bands are double occupied. Setting bands=(3,5) means including band index 3 and 4 which is the highest occupied band and the lowest unoccupied band.
"""

from gpaw.response.g0w0 import G0W0

gw = G0W0(calc='Si_groundstate.gpw', # student: calc='???_groundstate.gpw',
          nbands=30,  # student: nbands=???,  # number of bands for calculation of self-energy
          bands=(3, 5), # student: bands=(?, ?), # VB and CB
          ecut=20.0,  # student: ecut=???, # plane-wave cutoff for self-energy (20-200)
          integrate_gamma='WS',  # Use supercell Wigner-Seitz truncation for W.
          filename='Si-g0w0') # student: filename='???-g0w0'
gw.calculate()

# %%
"""
The dictionary is stored in ???-g0w0_results.pckl. From the dict it is for example possible to extract the direct bandgap at the Gamma point.
"""

import pickle

results = pickle.load(open('Si-g0w0_results_GW.pckl', 'rb')) # student: open('???-g0w0_results_GW.pckl', 'rb')
direct_gap = results['qp'][0, 0, -1] - results['qp'][0, 0, -2] 

print('Direct bandgap of Si:', direct_gap) # student: 'Direct bandgap of ???:', direct_gap

# %%
"""
Can we trust the calculated value of the direct bandgap? A check for convergence with respect to the plane wave cutoff energy and number of k points is necessary. This is done by changing the respective values in the groundstate calculation and restarting.

For more details on the convergence and other parameters you can look at this tutorial: https://gpaw.readthedocs.io/tutorialsexercises/opticalresponse/gw_tutorial/gw_tutorial.html#gw-tutorial

Typical convergence would require ecut=300 eV, 8x8x8 k-points and ecut_extrapolation=.True. for Carbon.

References:

   1. F. Hüser, T. Olsen, and K. S. Thygesen, “Quasiparticle GW calculations for solids, molecules, and two-dimensional materials”, Phys. Rev. B 87, 235132 (2013).
"""
