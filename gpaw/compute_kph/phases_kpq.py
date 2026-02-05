from gpaw import GPAW
import numpy as np
from gpaw.mpi import world
from gpaw.response.pw_parallelization import Blocks1D

calc_lcao = GPAW('scf.gpw', parallel={'domain': 1, 'band': 1})
calc_PW = GPAW('scf_PW.gpw', parallel={'domain': 1, 'band': 1})


def get_coefficients(psi_list, phi_list):
    psi_flat = [psi.ravel() for psi in psi_list]
    phi_flat = [phi.ravel() for phi in phi_list]    
    Psi = np.column_stack(psi_flat)
    Phi = np.column_stack(phi_flat)
    C, residuals, rank, s = np.linalg.lstsq(Phi, Psi, rcond=None)
    return C


nk = len(calc_PW.get_bz_k_points())
nbands = calc_PW.get_number_of_bands()
#all_bands_PW = list(range(12, 20))
all_bands_PW = list(range(3, 15))
all_bands_lcao = list(range(1, 30))
 
if world.rank == 0:
    print(f'{all_bands_PW = }', flush=True)
    print(f'{all_bands_lcao = }', flush=True)
    print(f'{nk = }', flush=True)
phases = []
for k in range(0, nk):
    phi_list = [calc_lcao.get_pseudo_wave_function(band=b, kpt=k, broadcast=True, periodic=True)
                       for b in all_bands_lcao]
    psi_list = [calc_PW.get_pseudo_wave_function(band=b, kpt=k, broadcast=True, periodic=True)
                       for b in all_bands_PW]
    
    
    C = get_coefficients(psi_list, phi_list)
    if world.rank == 0:
        print(f'{k = }', flush=True)
        phases.append({
                "k": k,
                "C": C,
                 })

import pickle
if world.rank == 0:
    with open("C_phases.pkl", "wb") as f:
        pickle.dump(phases, f)


