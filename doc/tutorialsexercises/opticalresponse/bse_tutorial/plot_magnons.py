import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

a = np.loadtxt('magnons_q.dat')
a[:, 1:] -= np.min(a[:, 1:])
a[:, 1:] *= 1000

plt.plot(a[:, 0], a[:, 1], '-o')
plt.plot(a[:, 0], a[:, 2], '-o')

plt.plot(2 * [a[4, 0]], [0, 1.1 * np.max(a[:, 2])], '-', c='0.5')
plt.xticks([a[0, 0], a[4, 0], a[-1, 0]], [r'$X$', r'$\Gamma$', r'$Y$'],
           size=16)
plt.yticks([0, 50, 100], [r'$0$', r'$50$', r'$100$'], size=16)
plt.ylabel(r'$\omega_q\;[\mathrm{meV}]$', size=22)
plt.axis([a[0, 0], a[-1, 0], 0, 1.1 * np.max(a[:, 2])])

plt.tight_layout()
# plt.show()
plt.savefig('magnons.png')
