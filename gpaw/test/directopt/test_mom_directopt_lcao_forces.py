import numpy as np
import pytest
from gpaw import GPAW, restart
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.directmin.tools import excite


@pytest.mark.mom
@pytest.mark.do
def test_mom_directopt_lcao_forces(in_tmp_dir, gpw_files):
    delta = 0.01
    calc = GPAW(gpw_files['co_mom_do_lcao_forces'])
    # XXX(rg): Remove hack after tchem-gl-13
    # calc.set_positions()
    # calc.wfs.eigensolver.initialize_dm_helper(calc.wfs,
    # calc.hamiltonian, calc.density, calc.log)
    atoms = calc.atoms
    atoms.calc = calc

    f_sn = excite(calc, 0, 0, spin=(0, 0))

    calc.set(eigensolver=LCAOETDM(excited_state=True,
                                  representation='u-invar',
                                  matrix_exp='egdecomp-u-invar'))
    prepare_mom_calculation(calc, atoms, f_sn)
    F = atoms.get_forces()

    # Test overlaps
    calc.wfs.occupations.initialize_reference_orbitals()
    for kpt in calc.wfs.kpt_u:
        f_n = calc.get_occupation_numbers(spin=kpt.s)
        P = calc.wfs.occupations.calculate_weights(kpt, 1.0)
        assert (np.allclose(P, f_n))

    calc.write('co.gpw', mode='all')

    # Exercise fixed occupations and no update of numbers in OccupationsMOM
    atoms, calc = restart('co.gpw', txt='-')
    e0 = atoms.get_potential_energy()
    for kpt in calc.wfs.kpt_u:
        f_sn[kpt.s] = kpt.f_n
    for i in [True, False]:
        prepare_mom_calculation(calc, atoms, f_sn,
                                use_fixed_occupations=i,
                                update_numbers=i)
        e1 = atoms.get_potential_energy()
        for spin in range(calc.get_number_of_spins()):
            if i:
                f_n = calc.get_occupation_numbers(spin=spin)
                assert (np.allclose(f_sn[spin], f_n))
            assert (np.allclose(f_sn[spin],
                                calc.wfs.occupations.numbers[spin]))
        assert e0 == pytest.approx(e1, abs=1e-2)

    E = []
    for i in [-1, 1]:
        atoms, calc = restart('co.gpw', txt='-')
        p = atoms.positions.copy()
        p[0, 2] -= delta / 2. * i
        p[1, 2] += delta / 2. * i
        atoms.set_positions(p)
        E.append(atoms.get_potential_energy())

    f = np.sqrt(((F[1, :] - F[0, :])**2).sum()) * 0.5
    fnum = (E[0] - E[1]) / (2. * delta)     # central difference

    print(fnum)
    assert fnum == pytest.approx(12.407162321236331, abs=0.01)
    assert f == pytest.approx(fnum, abs=0.1)
