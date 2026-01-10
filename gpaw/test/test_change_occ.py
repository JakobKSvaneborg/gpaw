import pytest
import numpy as np
from ase.build import molecule
from gpaw.dft import DFT


def test_occ():

    etot_fix = 22.407085
    fz = 21.22872
    forces_fix = np.array([[0, 0, fz], [0, 0, -fz]])

    atoms = molecule('H2', cell=[4, 4, 4])
    atoms.center()
    atoms.set_pbc(True)

    ppcg = {'name': 'ppcg',
            'niter': 30,
            'min_niter': 2}

    params = {'xc': 'PBE',
              'mode': {'name': 'pw', 'ecut': 400},
              'nbands': 3,
              'eigensolver': ppcg,
              'convergence': {'eigenstates': 1e-4,
                              'density': 1e-2,
                              'forces': 1e-1}}

    occ_fixed = {'name': 'fixed', 'numbers': [[0, 1, 0], [0, 1, 0]]}

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge()

    dft.change_occupations(occ_fixed)

    ase_calc = dft.ase_calculator()
    etot_occ = ase_calc.get_potential_energy(atoms)
    forces_occ = ase_calc.get_forces(atoms)

    if 0:
        from gpaw.new.ase_interface import GPAW
        params['occupations'] = occ_fixed
        calc = GPAW(**params)
        atoms.calc = calc
        etot_fix = atoms.get_potential_energy()
        forces_fix = atoms.get_forces()

    assert etot_occ == pytest.approx(etot_fix, abs=1e-4)
    assert forces_occ == pytest.approx(forces_fix, abs=5e-2)


if __name__ == "__main__":
    test_occ()
