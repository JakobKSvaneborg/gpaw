import pytest
import numpy as np
from ase.build import molecule
from gpaw.dft import DFT
from gpaw.new.ase_interface import GPAW
from gpaw.mpi import rank0_call


def test_changes():

    etot_hse = 23.546547
    fz = 21.11820
    forces_hse = np.array([[0, 0, fz], [0, 0, -fz]])

    atoms = molecule('H2', cell=[4, 4, 4])
    atoms.center()
    atoms.set_pbc(True)

    params = {'xc': 'PBE',
              'mode': {'name': 'pw', 'ecut': 400},
              'nbands': 3,
              'convergence': {'eigenstates': 1e-4,
                              'density': 1e-2,
                              'forces': 1e-3}}

    occ_fixed = {'name': 'fixed', 'numbers': [[0, 1, 0], [0, 1, 0]]}

    mixer = {'method': 'fullspin',
             'backend': 'fft',
             'beta': 0.05,
             'nmaxold': 7,
             'weight': 50.0}

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge()

    with pytest.raises(AssertionError):
        dft.change(xc='LDA')

    dft.change(xc='HSE06', eigensolver='ppcg', mixer=mixer,
               occupations=occ_fixed, convergence={'energy': 1e-2})

    ase_calc = dft.ase_calculator()
    etot_xc = ase_calc.get_potential_energy(atoms)
    forces_xc = ase_calc.get_forces(atoms)

    if 0:
        params['xc'] = 'HSE06'
        params['occupations'] = occ_fixed
        calc = GPAW(**params)
        atoms.calc = calc
        etot_hse = atoms.get_potential_energy()
        forces_hse = atoms.get_forces()

    assert etot_xc == pytest.approx(etot_hse, abs=1e-4)
    assert forces_xc == pytest.approx(forces_hse, abs=1e-2)

def test_gather():

    atoms = molecule('H2', cell=[4, 4, 4])
    atoms.center()
    atoms.set_pbc(True)

    params = {'xc': 'PBE',
              'mode': {'name': 'pw'},
              'nbands': 3,
              'convergence': {'eigenstates': 1e-4,
                              'density': 1e-4,
                              'forces': 1e-3}}

    # preconverge with PBE
    calc = GPAW(**params)
    atoms.calc = calc
    etot_test = atoms.get_potential_energy()
    forces_test = atoms.get_forces()

    # dft = DFT(atoms, **params)
    # dft.converge()
    # dft.energy()
    # dft.forces()
    # etot_test = dft.results['energy']
    # forces_test = dft.results['forces']

    calc.dft.gather_master()

    # ase_calc = dft.ase_calculator()
    # etot = ase_calc.get_potential_energy(atoms)
    # forces = ase_calc.get_forces(atoms)

    def get_energy_and_forces(calc, atoms):
        etot = calc.get_potential_energy(atoms)
        forces = calc.get_forces(atoms)
        return (etot, forces)

    etot, forces = rank0_call(get_energy_and_forces, calc.world)(calc, atoms)

    assert etot == pytest.approx(etot_test)
    assert forces == pytest.approx(forces_test)

if __name__ == "__main__":
    test_gather()
