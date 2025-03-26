# creates xas_h2s_spectrum.png
from gpaw.xas import XAS
from gpaw.utilities.folder import Folder

import matplotlib.pyplot as plt


dks = 164.623
so = 1.2  # SO splitting from experiment

dks_energies = [dks, dks + so]  # the dks energies for the two 2p edges
w_xas = [2 / 3, 1 / 3]  # weight distribution of the two 2p edges

xas = XAS.restart('me_h2s_xas.npz')

e_s, f_s_c = xas.get_oscillator_strength(dks=dks_energies, w=w_xas)
f_s = f_s_c.sum(0) / 3  # average over all dierctions

e, f = Folder(0.4, 'Lorentz').fold(e_s, f_s)

plt.plot(e, f)
plt.bar(e_s, f_s, width=0.05)
plt.xlabel('energy [eV]')
plt.ylabel('FOS [1/eV]')

plt.savefig('xas_h2s_spectrum.png')
