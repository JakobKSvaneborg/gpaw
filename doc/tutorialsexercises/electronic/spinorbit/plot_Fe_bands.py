# web-page: Fe_bands.png
import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates

calc = GPAW('Fe_bands.gpw', txt=None)

x = np.loadtxt('Fe_kpath.dat')
X = np.loadtxt('Fe_highsym.dat')

e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                   for k in range(len(calc.get_ibz_k_points()))]
                  for s in range(2)])
e_snk = np.swapaxes(e_skn, 1, 2)
ef = GPAW('Fe_gs.gpw').get_fermi_level()
e_snk -= ef
for e_k in e_snk[0]:
    plt.plot(x, e_k, '--', c='0.5')
for e_k in e_snk[1]:
    plt.plot(x, e_k, '--', c='0.5')

soc = soc_eigenstates(calc)
e_nk = soc.eigenvalues().T
s_knv = soc.spin_projections()

e_nk -= soc.fermi_level
s_nk = (s_knv[:, :, 2].T + 1.0) / 2.0

plt.xticks(X, [r'$\Gamma$', '(010)   H   (001)', r'$\Gamma$'], size=20)
plt.yticks(size=20)
for i in range(len(X))[1:-1]:
    plt.plot(2 * [X[i]], [1.1 * np.min(e_nk), 1.1 * np.max(e_nk)],
             c='0.5', linewidth=0.5)

plt.scatter(np.tile(x, len(e_nk)), e_nk.reshape(-1),
            c=s_nk.reshape(-1),
            s=5,
            marker='+')
plt.plot([0, x[-1]], 2 * [0.0], '-', c='0.5')

plt.ylabel(r'$\varepsilon_n(k)$ [eV]', size=24)
plt.axis([0, x[-1], -0.5, 0.5])
# pl.show()
plt.savefig('Fe_bands.png')
