import matplotlib.pyplot as plt
import numpy as np


chi_bsep = np.load('chi_MoS2_BSEPlus.npy')
chi_bse = np.load('chi_MoS2_BSE.npy')
chi_rpa = np.load('chi_MoS2_RPA.npy')
x = np.linspace(0, 50, 5001)

data_high_w = np.loadtxt('MoS2_q_0p060Ainv.csv', delimiter=',')
w_high = data_high_w[:, 0]
eels_high = data_high_w[:, 1]

data_low_w = np.loadtxt('MoS2_q_0p015invA.csv', delimiter=',')
w_low = data_low_w[:, 0]
eels_low = data_low_w[:, 1]

plt.plot(x, -chi_bsep[:, 0, 0].imag, label='BSE+')
plt.plot(x, -chi_bse[:, 0, 0].imag, label='BSE')
plt.plot(x, -chi_rpa[:, 0, 0].imag, label='RPA')
plt.plot(w_high, eels_high * 150, '.', color='black',
         label='Experimental data')

plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel(r'$\mathrm{I}_\mathrm{EELS}$' + ' [arb. units]')
plt.xlim(0, 30)
plt.ylim(0, 3)
plt.savefig('eels_MoS2.png')

plt.plot(w_low, eels_low / 3 + 0.08, '.', color='black',
         label='Experimental data')
plt.xlim(0, 4)
plt.ylim(0, 0.4)
plt.savefig('eels_MoS2_low_frequencies.png')
