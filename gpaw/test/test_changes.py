import pytest
import numpy as np
from ase.build import molecule
from gpaw.dft import DFT
from gpaw.new.ase_interface import GPAW


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
              'nbands': 4,
              # 'spinpol': True,
              # 'kpts': {'size': [3, 3, 3]},
              # 'parallel': {'domain': 1},
              'convergence': {'eigenstates': 1e-4,
                              'density': 1e-5,
                              'forces': 1e-3}}

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge()
    # in a.u.
    etot_test = dft.energy()
    forces_test = dft.forces()
    psit_nR_test = dft.wave_functions(n1=0, n2=1, kpt=0, spin=0)
    nt_sR_test = dft.densities().pseudo_densities().gather()
    # get_all_electron_density broken for domain_comm > 1
    n_sR_test = dft.densities().all_electron_densities().gather()

    newdft = dft.gather()

    if dft.comm.rank == 0:
        newdft.converge()   # SCF needed to set occupations
        etot = newdft.energy()
        forces = newdft.forces()
        psit_nR = newdft.wave_functions(n1=0, n2=1, kpt=0, spin=0)
        nt_sR = newdft.densities().pseudo_densities().data
        n_sR = newdft.densities().all_electron_densities().data

        # in a.u.
        assert etot == pytest.approx(etot_test)
        assert forces == pytest.approx(forces_test, abs=1e-3)
        assert psit_nR.data == pytest.approx(psit_nR_test.data, abs=1e-5)
        assert nt_sR == pytest.approx(nt_sR_test.data, abs=1e-5)
        if dft.density.nt_sR.desc.comm.size == 1:
            # get_all_electron_density broken for domain_comm > 1
            assert n_sR == pytest.approx(n_sR_test.data, abs=1e-5)
    else:
        assert newdft is None


if __name__ == "__main__":
    test_gather()
