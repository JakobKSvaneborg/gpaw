import matplotlib.pyplot as plt
import numpy as np


def refractive_index(chi):
    eps = 1 + np.pi * 4 * chi
    eps1 = eps.real
    eps2 = eps.imag
    N = np.sqrt(0.5 * (np.sqrt(eps1**2 + eps2**2) + eps1))
    return N


def eels(chi):
    eps = np.pi * 4 * chi + 1
    return (-eps**(-1)).imag


chi_bsep = np.load('chi_TiO2_BSEPlus.npy')
chi_bse = np.load('chi_TiO2_BSE.npy')
chi_rpa = np.load('chi_TiO2_RPA.npy')
x = np.linspace(0, 50, 5001)

exp_data = np.loadtxt('tio2_n_rutile_inplane.csv', delimiter=',')
freq = exp_data[:, 0]
exp_n = exp_data[:, 1]

n_bsep = refractive_index(-chi_bsep[:, 0, 0])
n_bse = refractive_index(-chi_bse[:, 0, 0])
n_rpa = refractive_index(-chi_rpa[:, 0, 0])

plt.plot(x, n_bsep, label='BSE+')
plt.plot(x, n_bse, label='BSE')
plt.plot(x, n_rpa, label='RPA')
plt.plot(freq, exp_n, '.', color='black', label='Experimental data')

plt.legend()
plt.xlabel(r'$\omega$' + ' [eV]')
plt.ylabel(r'$n(\omega)$')
plt.xlim(0, 6)
plt.ylim(0, 5)
plt.savefig('n_TiO2.png')
plt.close()

eels_bsep = eels(-chi_bsep[:, 0, 0])
eels_bse = eels(-chi_bse[:, 0, 0])
eels_rpa = eels(-chi_rpa[:, 0, 0])
exp_data = np.loadtxt('eels_tio2_rutile.csv', delimiter=',')
freq = exp_data[:, 0]
exp_eels = exp_data[:, 1]

plt.plot(x, eels_bsep, label='BSE+')
plt.plot(x, eels_bse, label='BSE')
plt.plot(x, eels_rpa, label='RPA')
plt.plot(freq, exp_eels * 40, '.', color='black', label='Experimental data')

plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel(r'$\mathrm{I}_\mathrm{EELS}$' + ' [arb. units]')
plt.xlim(0, 25)
plt.ylim(0, 1.2)
plt.savefig('eels_TiO2.png')
