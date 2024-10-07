# %%
"""
# Inclusion of excitonic effects and Discussion
"""

# %%
"""
Now, we will finally visualize the absorption spectra we have calculated. The script below is setup for plotting and saving the absorption spectra for BN with 24x24x1 k-points. First we load the x, y, and z components and plot all three curves in different colors.
"""

# %%
#Plot the absorption spectrum:
import numpy as np
import matplotlib.pyplot as plt
plt.figure()

#Of course you need to input the name you gave the files earlier own yourself and only plot the components you calculated.
#Here is only showed hot to plot the x-component.

absox = np.loadtxt('Si_rpa_x.csv', delimiter=',') # student: absox = np.loadtxt('???_rpa_x.csv', delimiter=',')
plt.plot(absox[:, 0], absox[:, 4], label='RPA_x Si 12x12x4', lw=2, color='b')

plt.xlabel(r'$\hbar\omega\;[eV]$', size=20)
plt.ylabel(r'$Im(\epsilon)$', size=20)
plt.xticks(size=16)
plt.yticks(size=16)
plt.tight_layout()
plt.axis([0.0, 10.0, None, None])
plt.legend()

#plt.show()
plt.savefig('rpa_Si.png')

# %%
"""
Now open the saved figure and inspect the absorption spectra. Does it look as you expected? Can you guess the band gap from the absorption spectra and which of the band gaps calculated on day 2 does it match?

Now plot the absorption spectra for the other k-point meshes you calculated and compare. You could for instance modify the script above to plot, in different colors, all spectra in the same plot for comparison. Would you say that the calculation is converged?

Finally, talk with the other groups and compare your absorption spectra with their absorption spectra. Is the same number of k-points needed to obtain same degree of convergence for the different materials? Is there any difference for the 2D material (Boron-Nitride)?

For a more realistic description one would have to include excitonic effects (i.e. the electron-hole interaction). 
In the next example, we will calculate the absorption spectrum using the Bethe-Salpeter equation (BSE). For theoretical understanding you can read the brief description provided as in the link below: https://gpaw.readthedocs.io/documentation/bse/bse.html

We start by calculating the ground state density and diagonalizing the resulting Hamiltonian. The last line in the script creates a .gpw file which contains all the informations of the system, including the wavefunctions.
"""

from ase import Atoms
from ase.build import bulk
from gpaw import GPAW
from gpaw import PW
from gpaw.occupations import FermiDirac

Si = bulk('Si', 'diamond', a=5.4) # student:
atoms = Si # student: atoms = ???

calc = GPAW(mode=PW(400),  # student: mode=PW(???),
            xc='PBE',
            occupations=FermiDirac(width=0.001),
            parallel={'domain': 1, 'band': 1},
            kpts={'size': (8, 8, 8), 'gamma': True}, # student: kpts={'size': (?, ?, ?), 'gamma': True},
            txt='gs_Si.txt') # student: txt='gs_???.txt'

atoms.calc = calc
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian(nbands=100) # student: nbands=???
calc.write('gs_Si.gpw', mode='all') # student: calc.write('gs_???.gpw', mode='all')


# %%
"""
Below we will set up the Bethe-Salpeter Hamiltonian in a basis of the ?? valence bands and ?? conduction bands.
However, the screened interaction that enters the Hamiltonian needs to be converged with respect the number of unoccupied bands. Next we calculate the dynamical dielectric function using the Bethe-Salpeter equation. The imaginary part is proportional to the absorption spectrum. We will calculate the dielectric function within the Random Phase Approximation (with the same convergence parameters for comparison).
"""


import numpy as np
from gpaw.response.bse import BSE
from gpaw.response.df import DielectricFunction

ecut = 50 # student: ecut = ???
eta = 0.2

df = DielectricFunction('gs_Si.gpw', # student: 'gs_???.gpw',
                        ecut=ecut,
                        frequencies=np.linspace(0, 10, 1001),
                        nbands=8, # student: nbands=??,
                        intraband=False,
                        hilbert=False,
                        eta=eta,
                        txt='rpa_Si.txt') # student: txt='rpa_???.txt'

df.get_dielectric_function(filename='eps_rpa_Si.csv') # student: filename='eps_rpa_???.csv'

bse = BSE('gs_Si.gpw', # student: 'gs_???.gpw',
          ecut=ecut,
          valence_bands=range(0, 4), # student: valence_bands=range(?, ?),
          conduction_bands=range(4, 8), # student: conduction_bands=range(?, ?),
          nbands=50, # student: nbands=???,
          mode='BSE',
          integrate_gamma='sphere',
          txt='bse_Si.txt') # student: txt='bse_???.txt'

bse.get_dielectric_function(filename='eps_bse_Si.csv', # student: filename='eps_bse_???.csv',
                            eta=eta,
                            write_eig='bse_Si_eig.dat', # student: write_eig='bse_???_eig.dat',
                            w_w=np.linspace(0.0, 10.0, 10001))



# %%
"""
Note the .csv output files that contains the spectre. The script also generates a .dat file that contains the
BSE eigenvalues. The spectrum essentially consists of a number of peaks centered on the eigenvalues. Below we will
plot it along with the RPA spectrum for comparison.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.figure()

a = np.loadtxt('eps_rpa_Si.csv', delimiter=',') # student: a = np.loadtxt('eps_rpa_???.csv', delimiter=',')
plt.plot(a[:, 0], a[:, 4], label='RPA', lw=2)

a = np.loadtxt('eps_bse_Si.csv', delimiter=',') # student: a = np.loadtxt('eps_bse_???.csv', delimiter=',')
plt.plot(a[:, 0], a[:, 2], label='BSE', lw=2)

plt.xlabel(r'$\hbar\omega\;[eV]$', size=24)
plt.ylabel(r'$\epsilon_2$', size=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.axis([2.0, 6.0, None, None])
plt.legend()

# plt.show()
plt.savefig('bse_Si.png') # student: plt.savefig('bse_???.png')

# %%
"""
Note: The parameters that need to be converged in the calculation are the k-points in the initial ground state calculation. In addition the following keywords in the BSE object should be converged: the plane wave cutoff, the numbers of bands used to calculate the screened interaction, the list of valence bands and the list of conduction bands included in the Hamiltonian. You can find an example calculation for Silicon in the link below: https://gpaw.readthedocs.io/tutorialsexercises/opticalresponse/bse_tutorial/bse_tutorial.html
"""
