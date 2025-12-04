from ase.io.jsonio import write_json
from gpaw import GPAW
from gpaw.defects import charged_defect_corrections
from pathlib import Path

charge = -3
epsilon = 12.7
def_idx = 0
corrected = []
uncorrected = []
repeats = [1, 2, 3, 4]
for N in repeats:
    label = f'GaAs_{N}x{N}x{N}'
    prs_path = Path(f'{label}_prs.gpw')
    def_path = Path(f'{label}_def.gpw')

    calc_prs = GPAW(prs_path)
    calc_def = GPAW(def_path)

    elc = charged_defect_corrections(calc_pristine=calc_prs,
                                     calc_defect=calc_def,
                                     defect_index=def_idx,
                                     charge=charge,
                                     epsilon=epsilon)
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
