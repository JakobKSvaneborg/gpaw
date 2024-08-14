# web-page: C_GW_k8_extrap.png
import pickle
import numpy as np
from ase.parallel import paropen
import matplotlib.pyplot as plt

plt.figure(figsize=(6.5, 4.5))

ecuts = np.array([100, 200, 300, 400])
color = ['ro-', 'bo-', 'go-', 'ko-', 'co-', 'mo-', 'yo-']
direct_gap = np.zeros(4)
extrapolated_direct_gap = np.zeros(4)


def get_gamma_gap(fname):
    data = pickle.load(paropen(fname, 'rb'))
    return data['qp'][0, 0, 1] - data['qp'][0, 0, 0]


k = 8
for i, ecut in enumerate([100, 200, 300, 400]):
    fname = f'C-g0w0_k{k}_ecut{ecut}_results_GW.pckl'
    direct_gap[i] = get_gamma_gap(fname)

    fname = f'C-g0w0_k{k}_ecut_{ecut}_automatic_extrapolate_results_GW.pckl'
    extrapolated_direct_gap[i] = get_gamma_gap(fname)

plt.plot(1 / (ecuts**(3. / 2.)), direct_gap, 'ko-',
         label=f'({k}x{k}x{k}) non-extrapolated')

plt.plot(1 / (ecuts**(3. / 2.)), extrapolated_direct_gap, 'rx-',
         label=f'({k}x{k}x{k}) extrapolated')

extrap_gap, slope = np.linalg.solve([[1, 1. / 300.**(3. / 2)],
                                     [1, 1. / 400.**(3. / 2)]],
                                    [direct_gap[2], direct_gap[3]])
xs = np.linspace(0, 1 / 400.**(3. / 2), 1000)
plt.plot(xs, extrap_gap + slope * xs, 'k--')

plt.xticks([1. / 100**(3. / 2), 1. / 200**(3. / 2), 1. / 400**(3. / 2), 0],
           [100, 200, 400, r'$\infty$'])
plt.xlabel('Cutoff energy (eV)')
plt.ylabel('Direct band gap (eV)')
plt.title('non-selfconsistent G0W0@LDA')
plt.legend(loc='upper right')
plt.savefig('C_GW_k8_extrap_automatic.png')
plt.show()
