import pytest
import numpy as np
from ase.build import molecule
from gpaw.dft import DFT


def test_xc():

    etot_hse = -9.773301
    fz = 1.44491
    forces_hse = np.array([[0, 0, fz], [0, 0, -fz]])

    atoms = molecule('H2', cell=[4, 4, 4])
    atoms.center()
    atoms.set_pbc(True)

    params = {'xc': 'PBE',
              'mode': {'name': 'pw', 'ecut': 400},
              'nbands': 3,
              'convergence': {'eigenstates': 1e-4,
                              'density': 1e-2,
                              'forces': 1e-2}}

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge()

    dft.change_xc('HSE06')

    ase_calc = dft.ase_calculator()
    etot_xc = ase_calc.get_potential_energy(atoms)
    forces_xc = ase_calc.get_forces(atoms)

    if 0:
        from gpaw.new.ase_interface import GPAW
        # check against HSE
        params['xc'] = 'HSE06'
        calc = GPAW(**params)
        atoms.calc = calc
        etot_hse = atoms.get_potential_energy()
        forces_hse = atoms.get_forces()

    assert etot_xc == pytest.approx(etot_hse, abs=1e-4)
    assert forces_xc == pytest.approx(forces_hse, abs=5e-2)


if __name__ == "__main__":
    test_xc()
