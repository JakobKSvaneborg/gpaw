import numpy as np
from ase.io.jsonio import write_json
from gpaw import GPAW
from gpaw.defects import ElectrostaticCorrections
from pathlib import Path

sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
charge = -3
epsilon = 12.7
corrected = []
uncorrected = []
profiles = []
repeats = [1, 2, 3]
for N in repeats:
    label = f'GaAs_{N}x{N}x{N}'
    prs_path = Path(f'{label}_prs.gpw')
    def_path = Path(f'{label}_def.gpw')

    pristine = GPAW(prs_path).get_atoms()

    # defect position
    r0 = pristine.positions[0, :]

    elc = ElectrostaticCorrections(pristine=prs_path,
                                   defect=def_path,
                                   r0=r0,
                                   charge=charge,
                                   sigma=sigma,
                                   epsilon=epsilon,
                                   method='full-planar')
    E_corr = elc.calculate_corrected_formation_energy()
    E_uncorr = elc.calculate_uncorrected_formation_energy()
    profile = elc.calculate_potential_profile()

    corrected.append(E_corr)
    uncorrected.append(E_uncorr)
    profiles.append(profile)

res = {'repeats': repeats, 'corrected': corrected,
       'uncorrected': uncorrected, 'profiles': profiles}

write_json('electrostatics.json', res)
