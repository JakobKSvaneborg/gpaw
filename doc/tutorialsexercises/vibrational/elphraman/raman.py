import numpy as np

from gpaw.old.calculator import GPAW
from gpaw.elph import ResonantRamanCalculator


calc = GPAW("scf.gpw", parallel={'domain': 1, 'band': 1})
wph_w = np.load("vib_frequencies.npy")

rrc = ResonantRamanCalculator(calc, wph_w)
rrc.calculate_raman_tensor(rrc.nm_to_eV(488.))
