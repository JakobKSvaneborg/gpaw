import matplotlib.pyplot as plt
import numpy as np


chi_bsep = np.load('chi_MoS2_BSE_plus.npy')
chi_bse = np.load('chi_MoS2_BSE.npy')
chi_rpa = np.load('chi_MoS2_RPA.npy')
x = np.linspace(0, 50, 5001)

plt.plot(x, -chi_bsep[:, 0, 0].imag, label='BSE+')
plt.plot(x, -chi_bse[:, 0, 0].imag, label='BSE')
plt.plot(x, -chi_rpa[:, 0, 0].imag, label='RPA')

plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel(r'$\mathrm{I}_\mathrm{EELS}$' + ' [arb. units]')
plt.xlim(0, 30)
plt.ylim(0, 3)
plt.savefig('eels_MoS2.png')

plt.xlim(0, 4)
plt.ylim(0, 0.6)
plt.savefig('eels_MoS2_low_frequencies.png')
