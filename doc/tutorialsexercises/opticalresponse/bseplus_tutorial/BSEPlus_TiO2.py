from gpaw import GPAW
import numpy as np
from gpaw.response.bse import BSEPlus
from ase.dft.bandgap import bandgap

calc_bse = 'fixed_density_calc_TiO2_bse.gpw'
calc_rpa = 'fixed_density_calc_TiO2_rpa.gpw'

ecut = 80
eta = 0.1
q_c = [0.0, 0.0, 0.0]
bse_valence_bands = range(18, 24)
bse_conduction_bands = range(24, 30)
bse_nbands = 60
rpa_nbands = 130
w_w = np.linspace(0, 50, 5001)

gap, _, _ = bandgap(GPAW(calc_rpa), direct=True)
eshift = 3.3 - gap

bseplus = BSEPlus(bse_gpw=calc_bse,
                    bse_valence_bands=bse_valence_bands,
                    bse_conduction_bands=bse_conduction_bands,
                    bse_nbands=bse_nbands,
                    rpa_gpw=calc_rpa,
                    rpa_nbands=rpa_nbands,
                    w_w=w_w,
                    eshift=eshift,
                    eta=eta,
                    q_c=q_c,
                    ecut=ecut)

bseplus.get_chi_wGG(optical=True,
                     chi_BSE=True,
                     chi_RPA=True,
                     bsep_name='chi_TiO2_BSEPlus',
                     bse_name='chi_TiO2_BSE',
                     rpa_name='chi_TiO2_RPA')
