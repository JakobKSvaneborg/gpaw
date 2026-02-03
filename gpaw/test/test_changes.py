import pytest
import numpy as np
from ase.build import molecule
from gpaw.dft import DFT
from gpaw.mpi import rank0_call
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

    tol = {'etot': 1e-8, 'forces': 1e-3, 'density': 1e-5}

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge()

    res = {}
    res['etot'] = dft.energy()
    res['forces'] = dft.forces()
    res['psit_nR'] = dft.wave_functions(n1=0, n2=1, kpt=0, spin=0)
    res['nt_sR'] = dft.densities().pseudo_densities().gather()
    # get_all_electron_density broken for domain_comm > 1
    res['n_sR'] = dft.densities().all_electron_densities().gather()

    newdft = dft.gather()

    if dft.comm.rank == 0:
        # remove results for test to force recalculate
        newdft.results = {}
        # XXX Remove converge() once occupations are copied as well
        newdft.converge()   # SCF needed to set occupations
        new = {}
        new['etot'] = newdft.energy()
        new['forces'] = newdft.forces()
        new['psit_nR'] = newdft.wave_functions(n1=0, n2=1, kpt=0, spin=0)
        new['nt_sR'] = newdft.densities().pseudo_densities()
        new['n_sR'] = newdft.densities().all_electron_densities()

        for key in ['etot', 'forces']:
            assert new[key] == pytest.approx(res[key], abs=tol[key])

        dtol = tol['density']

        # wavefunction defined up to a phase (sign)
        psi_o = res['psit_nR'].data
        psi_n = new['psit_nR'].data
        idx = np.flatnonzero(np.abs(psi_n) > dtol)[0]
        sign = np.sign(psi_o.flat[idx] / psi_n.flat[idx])
        assert sign * psi_n == pytest.approx(psi_o, abs=dtol)

        for key in ['nt_sR', 'n_sR']:
            if dft.density.nt_sR.desc.comm.size > 1 and key == 'n_sR':
                # get_all_electron_density broken for domain_comm > 1
                continue
            assert new[key].data == pytest.approx(res[key].data, abs=dtol)
    else:
        assert newdft is None


if __name__ == "__main__":
    test_gather()
