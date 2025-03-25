from gpaw import GPAW, setup_paths
from gpaw.xas import XAS
from gpaw.utilities.folder import Folder

import matplotlib.pyplot as plt


setup_paths.insert(0, '.')

dks = 164.623
so = 1.2  # SO splitting form experiment

dks_energies = [dks, dks + so]  # the dks energyes for the two 2p edges
w_xas = [2 / 3, 1 / 3]  # wight distrubution of the two 2p edges

calc = GPAW('h2s_xas.gpw')

xas = XAS(calc)

e_s, f_s_c = xas.get_oscillator_strength(dks=dks_energies, w=w_xas)

f_s = f_s_c.sum(0) / 3

e, f = Folder(0.2, 'Lorentz').fold(e_s, f_s)

plt.plot(e, f)
plt.bar(e_s, f_s, width=0.05)

plt.savefig('xas_h2s_spectrum.png')
