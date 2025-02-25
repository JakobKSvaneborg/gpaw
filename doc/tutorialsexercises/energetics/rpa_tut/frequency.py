import numpy as np
from ase.parallel import paropen

from gpaw.xc.rpa import RPACorrelation

dw = 0.5
frequencies = np.array([dw * i for i in range(200)])
weights = dw * np.ones(len(frequencies))
weights[0] /= 2
weights[-1] /= 2

rpa = RPACorrelation('N2.gpw',
                     txt='frequency_equidistant.txt',
                     frequencies=frequencies,
                     weights=weights,
                     ecut=[50])
data = rpa.calculate_all_contributions()
Es_w = data.energy_wi[:, 0]

with paropen('frequency_equidistant.dat', 'w') as fd:
    for w, E in zip(frequencies, Es_w):
        print(w, E, file=fd)
