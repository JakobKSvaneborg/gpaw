from gpaw import GPAW, setup_paths
from gpaw.xas import XAS
import matplotlib.pyplot as plt

setup_paths.insert(0, '.')

dks_energy = 532.774  # from dks calcualtion

calc = GPAW('h2o_xas.gpw', legacy_gpaw=True)

xas = XAS(calc, mode='xas')
x, y = xas.get_spectra(fwhm=0.5, linbroad=[1.5, 536, 540], dks=dks_energy)
x_s, y_s = xas.get_spectra(stick=True, dks=dks_energy)

y_av = (y[0] + y[1] + y[2]) / 3
y_av_s = (y_s[0] + y_s[1] + y_s[2]) / 3

plt.plot(x, y_av)
plt.bar(x_s, y_av_s, width=0.05)
plt.savefig('xas_h2o_spectrum.png')
