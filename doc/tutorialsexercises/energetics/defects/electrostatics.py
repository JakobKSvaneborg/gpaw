import numpy as np
from ase.io.jsonio import write_json
from gpaw import GPAW
from gpaw.defects import ElectrostaticCorrections
from gpaw.defects.electrostatic import gather_electrostatic_potential
from pathlib import Path

sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
charge = -3
epsilon = 12.7
corrected = []
uncorrected = []
repeats = [1, 2, 3, 4]
for N in repeats:
    label = f'GaAs_{N}x{N}x{N}'
    prs_path = Path(f'{label}_prs.gpw')
    def_path = Path(f'{label}_def.gpw')

    calc_prs = GPAW(prs_path)
    calc_def = GPAW(def_path)

    atoms_prs = calc_prs.get_atoms()
    rvR_prs, phi_prs = gather_electrostatic_potential(calc_prs)
    rvR_def, phi_def = gather_electrostatic_potential(calc_def)

    # defect position
    r0 = atoms_prs.positions[0, :]

    elc = ElectrostaticCorrections(atoms_prs=atoms_prs,
                                   rphi_prs=(rvR_prs, phi_prs),
                                   rphi_def=(rvR_def, phi_def),
                                   r0=r0,
                                   charge=charge,
                                   sigma=sigma,
                                   epsilon=epsilon,
                                   method='full-planar')
    E_fnv = elc.calculate_correction()

    E_0 = calc_prs.get_potential_energy()
    E_X = calc_def.get_potential_energy()
    E_uncorr = E_X - E_0
    E_corr = E_uncorr + E_fnv

    if N == 2:
        profile = elc.calculate_potential_profile()

    corrected.append(E_corr)
    uncorrected.append(E_uncorr)

res = {'repeats': repeats, 'corrected': corrected,
       'uncorrected': uncorrected, 'profile': profile}

write_json('electrostatics.json', res)
