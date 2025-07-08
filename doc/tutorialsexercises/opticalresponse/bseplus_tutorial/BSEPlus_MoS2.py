from gpaw import GPAW
from gpaw.response.bse import BSEPlus
import numpy as np
from ase.dft.bandgap import bandgap


calc_bse = 'fixed_density_calc_MoS2_bse.gpw'
calc_rpa = 'fixed_density_calc_MoS2_rpa.gpw'

ecut = 50
eta = 0.05
q_c = [0.07407407, 0.0, 0.0]
bse_valence_bands = range(18, 26)
bse_conduction_bands = range(26, 34)
bse_nbands = 100
rpa_nbands = 170
w_w = np.linspace(0, 50, 5001)

gap, _, _ = bandgap(GPAW(calc_rpa), direct=True)
eshift = 2.53 - gap  # G0W0 bandgap according to C2DB

bseplus = BSEPlus(bse_gpw=calc_bse,
                  bse_valence_bands=bse_valence_bands,
                  bse_conduction_bands=bse_conduction_bands,
                  bse_nbands=bse_nbands,
                  rpa_gpw=calc_rpa,
                  rpa_nbands=rpa_nbands,
                  w_w=w_w,
                  eshift=eshift,
                  bse_add_soc=True,
                  eta=eta,
                  q_c=q_c,
                  truncation='2D',
                  ecut=ecut)

bseplus.calculate_chi_wGG(optical=False, chi_BSE=True, chi_RPA=True,
                          bsep_name='chi_MoS2_BSEPlus',
                          save_chi_BSE='chi_MoS2_BSE',
                          save_chi_RPA='chi_MoS2_RPA')
