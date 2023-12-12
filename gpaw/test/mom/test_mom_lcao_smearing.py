import pytest
from ase.build import molecule
from gpaw import GPAW, restart


@pytest.mark.mom
def test_mom_lcao_smearing(in_tmp_dir):
    atoms = molecule('CO')
    atoms.center(vacuum=2)

    calc = GPAW(mode='lcao',
                basis='dzp',
                nbands=7,
                h=0.24,
                xc='PBE',
                spinpol=True,
                symmetry='off',
                convergence={'energy': 100,
                             'density': 1e-3})

    atoms.calc = calc
    # Ground-state calculation
    E_gs = atoms.get_potential_energy()

    calc.write('co_lcao_gs.gpw', 'all')

    f_sn = []
    for spin in range(calc.get_number_of_spins()):
        f_n = calc.get_occupation_numbers(spin=spin)
        f_sn.append(f_n)

    ne0_gs = f_sn[0].sum()
    f_sn[0][3] -= 1.
    f_sn[0][5] += 1.

    # Test both MOM and fixed occupations with Gaussian smearing
    for i in [True, False]:
        atoms, calc = restart('co_lcao_gs.gpw', txt='-')

        # Excited-state calculation with Gaussian
        # smearing of the occupation numbers
        calc.set(occupations={'name': 'mom', 'numbers': f_sn,
                              'width': 0.01, 'use_fixed_occupations': i})
        E_es = atoms.get_potential_energy()

        f_n0 = calc.get_occupation_numbers(spin=0)
        ne0_es = f_n0.sum()

        dE = E_es - E_gs
        dne0 = ne0_es - ne0_gs
        print(dE)
        assert dE == pytest.approx(9.8445603513, abs=0.01)
        assert dne0 == pytest.approx(0.0, abs=1e-16)
