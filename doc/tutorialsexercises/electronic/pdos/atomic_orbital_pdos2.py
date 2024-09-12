import numpy as np
from gpaw import GPAW

doscalc = GPAW('au.gpw').dos()
energy = np.linspace(-10, 10, 201)
pdos = doscalc.raw_pdos(energy, a=0, l=2)
