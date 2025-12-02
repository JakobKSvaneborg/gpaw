import numpy as np
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE
import os

def run_bse_test():
    # Only verify we can run if Si_test.gpw exists, else run full

    # Force regeneration of gpw to ensure consistent bands
    if os.path.exists('Si_test.gpw'):
        os.remove('Si_test.gpw')

    a = 5.431
    atoms = bulk('Si', 'diamond', a=a)
    atoms.positions -= a / 8

    calc = GPAW(mode='pw',
                kpts={'size': (2, 2, 2), 'gamma': True},
                occupations=FermiDirac(0.001),
                nbands=8,
                convergence={'bands': -2, 'density': 1e-4})

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si_test.gpw', 'all')

    eshift = 0.8
    # Silicon has 4 valence electrons per atom -> 8 valence electrons total -> 4 occupied bands (0,1,2,3).
    # Valence bands: range(4) covers 0,1,2,3.
    # Conduction bands: range(4,8) covers 4,5,6,7.

    bse = BSE('Si_test.gpw',
              ecut=30.,
              valence_bands=range(4),
              conduction_bands=range(4, 8),
              eshift=eshift,
              nbands=8)

    print("Diagonalizing BSE...")
    # Just diagonalize indirectly via get_vchi or similar
    bse.get_dielectric_function(eta=0.2, w_w=np.linspace(0, 10, 2))

    if hasattr(bse, 'eig_data'):
        w_T = bse.eig_data[0]
        print(f"Number of eigenvalues: {len(w_T)}")

        # Calculate corrections for first 3 eigenvalues
        indices = [0, 1, 2]
        print(f"Calculating corrections for indices: {indices}")
        corrections = bse.calculate_perturbation_correction(indices)

        print("\nResults:")
        for i, idx in enumerate(indices):
            eig_val = w_T[idx] * 27.2114 # eV
            corr = corrections[i] * 27.2114 # eV
            print(f"Eigenvalue {idx}: {eig_val:.4f} eV")
            print(f"Correction  {idx}: {corr.real:.6f} + {corr.imag:.6f}j eV")

        # Check that eigenvalues are reasonable (Si gap is ~1.1 eV, + scissor 0.8 -> ~2 eV + binding energy reduction)
        # With coarse grid, it might be different, but clearly > 0.
        if abs(w_T[indices[0]] * 27.2114) > 0.1:
            print("PASS: Eigenvalues are non-zero.")
        else:
            print(f"FAIL: Eigenvalues are close to zero: {w_T[indices[0]] * 27.2114}")

    else:
        print("eig_data not found in bse object")

if __name__ == '__main__':
    run_bse_test()
