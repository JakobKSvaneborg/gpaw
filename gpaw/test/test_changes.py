import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from gpaw.dft import DFT
from gpaw.mpi import rank0_call
from gpaw.new.ase_interface import GPAW


def test_changes():

    etot_hse = -9.773299
    fz = 1.44907
    forces_hse = np.array([[0, 0, fz], [0, 0, -fz]])

    atoms = molecule('H2', cell=[4, 4, 4])
    atoms.center()
    atoms.set_pbc(True)

    params = {'xc': 'PBE',
              'mode': {'name': 'pw', 'ecut': 400},
              'nbands': 3,
              'convergence': {'eigenstates': 1e-4,
                              'density': 1e-2,
                              'forces': 1e-4}}

    # occ_fixed = {'name': 'fixed', 'numbers': [[0, 1, 0], [0, 1, 0]]}
    occ_fixed = {'name': 'fixed', 'numbers': [[1, 0, 0], [1, 0, 0]]}

    mixer = {'method': 'fullspin',
             'backend': 'fft',
             'beta': 0.05,
             'nmaxold': 7,
             'weight': 50.0}

    # diropt = {'name': 'etdm-fdpw', 'converge_unocc': True}
    # nomixer = {'backend': 'no-mixing'}
    xc = 'HSE06'

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge()

    with pytest.raises(AssertionError):
        dft.change(xc='LDA')

    dft.change(xc=xc, eigensolver='davidson', mixer=mixer,
               occupations=occ_fixed, convergence={'energy': 1e-2})

    ase_calc = dft.ase_calculator()
    etot_xc = ase_calc.get_potential_energy(atoms)
    forces_xc = ase_calc.get_forces(atoms)

    if 0:
        params['xc'] = xc
        params['occupations'] = occ_fixed
        # params['eigensolver'] = diropt
        # params['mixer'] = nomixer
        calc = GPAW(**params)
        atoms.calc = calc
        etot_hse = atoms.get_potential_energy()
        forces_hse = atoms.get_forces()

    assert etot_xc == pytest.approx(etot_hse, abs=1e-4)
    assert forces_xc == pytest.approx(forces_hse, abs=1e-2)


@pytest.mark.parametrize('mode', ['pw', 'fd'])
def test_gather(mode):

    atoms = molecule('H2', cell=[3, 3, 3])
    atoms.center()
    atoms.set_pbc(True)

    params = {'xc': 'PBE',
              'mode': {'name': mode}}

    tol = {'etot': 1e-8, 'forces': 1e-2, 'density': 1e-5}

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge(steps=2)

    ref = {}
    ref['etot'] = dft.calculate_energy()
    ref['forces'] = dft.calculate_forces()
    ref['psit_nR'] = dft.wave_functions(n1=0, n2=1, kpt=0, spin=0)
    ref['nt_sR'] = dft.densities().pseudo_densities().gather()
    # get_all_electron_density broken for domain_comm > 1
    ref['n_sR'] = dft.densities().all_electron_densities().gather()

    newdft = dft.gather()

    def compare_gathered(newdft, ref, tol):
        # remove results for test to enforce recalculate
        newdft.results = {}
        new = {}
        new['etot'] = newdft.calculate_energy()
        new['forces'] = newdft.calculate_forces()
        new['psit_nR'] = newdft.wave_functions(n1=0, n2=1, kpt=0, spin=0)
        new['nt_sR'] = newdft.densities().pseudo_densities()
        new['n_sR'] = newdft.densities().all_electron_densities()

        for key in ['etot', 'forces']:
            assert new[key] == pytest.approx(ref[key], abs=tol[key])

        dtol = tol['density']

        # wavefunction defined up to a global phase (sign)
        psi_o = ref['psit_nR'].data
        psi_n = new['psit_nR'].data
        idx = np.flatnonzero(np.abs(psi_n) > dtol)[0]
        sign = np.sign(psi_o.flat[idx] / psi_n.flat[idx])
        assert sign * psi_n == pytest.approx(psi_o, abs=dtol)

        for key in ['nt_sR', 'n_sR']:
            assert new[key].data == pytest.approx(ref[key].data, abs=dtol)
        return

    # only call on master and broadcast exceptions
    rank0_call(compare_gathered, dft.comm)(newdft, ref, tol)


@pytest.mark.parametrize('mode', ['pw', 'fd'])
def test_lcao_to_x(mode):
    atoms = Atoms('H', magmoms=[1])
    atoms.center(vacuum=1.5)

    dft = DFT(atoms, mode='lcao', symmetry='off')
    dft.converge()

    dft.change_mode(mode)
    dft.converge()

    atoms.positions[:] += 0.1
    dft.move_atoms(atoms)
    dft.converge()
    e1 = dft.calculate_energy()

    dft = DFT(atoms, mode=mode)
    dft.converge()
    e2 = dft.calculate_energy()
    assert e1 == pytest.approx(e2)


if __name__ == '__main__':
    test_lcao_to_x('pw')
