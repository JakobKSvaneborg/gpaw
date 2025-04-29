import pytest
from ase.units import Hartree, Bohr
import numpy as np
from ase.calculators.dftd3 import PureDFTD3

@pytest.mark.parametrize('parallel', [(1, 1), (1, 2), (2, 2), (2, 1)])
@pytest.mark.parametrize('mode', [{'name': 'pw', 'ecut': 400}, 'fd', 'lcao'])
def test_d3_extensions(mode, parallel, in_tmp_dir):
    from gpaw.new.ase_interface import GPAW
    from gpaw import restart
    from gpaw.mpi import world
    from gpaw.new.extensions import D3
    domain, band = parallel
    if world.size < domain * band:
        pytest.skip('Not enough cores for this test.')
    if world.size > domain * band * 2:
        pytest.skip('Too many cores for this test.')

    # 1. Create a calculation with a particular list of extensions.
    def get_atoms():
        from ase.build import molecule
        atoms = molecule('H2')
        atoms.center(vacuum=4)
        atoms.set_pbc((True, True, True))
        return atoms

    def D3ref(atoms):
        atoms = atoms.copy()
        atoms.calc = PureDFTD3(xc='PBE')
        return atoms.get_potential_energy(), atoms.get_forces()

    atoms = get_atoms()

    def get_calc(atoms):
        # To test multiple extensions, create two sprigs which add
        # up to k=ktot, which is what is tested in this test
        calc = GPAW(extensions=[D3(xc='PBE')],
                    symmetry='off',
                    parallel={'band': band, 'domain': domain},
                    kpts=(2, 1, 1),
                    mode=mode)
        atoms.calc = calc
        return calc

    calc = get_calc(atoms)

    E, F = atoms.get_potential_energy(), atoms.get_forces()
    D3_E, D3_F = D3ref(atoms)

    # Write the GPW file for the restart test later on (4.)
    print('Wrote the potential energy', E)
    calc.write('calc.gpw')

    # 2. Test that moving the atoms works after an SFC convergence
    atoms.positions[0, 2] -= 0.1
    movedE, movedF = atoms.get_potential_energy(), atoms.get_forces()

    movedD3_E, movedD3_F = D3ref(atoms)
    # Reset atoms to their original positions
    atoms.positions[0, 2] += 0.1

    # 3. Calculate a reference result without extensions
    calc = GPAW(mode=mode,
                kpts=(2, 1, 1),
                symmetry='off')
    atoms.calc = calc

    E0, F0 = atoms.get_potential_energy(), atoms.get_forces()

    # Manually evaluate the spring energy, and compare forces
    l = atoms.get_distance(0, 1)
    assert E == pytest.approx(E0 + D3_E)
    assert F == pytest.approx(F0 + D3_F)

    # Evaluate the reference energy and forces also for the moved atoms
    atoms.positions[0, 2] -= 0.1
    movedE0, movedF0 = atoms.get_potential_energy(), atoms.get_forces()
    l = atoms.get_distance(0, 1)
    assert movedE == pytest.approx(movedE0 + movedD3_E)
    assert movedF == pytest.approx(movedF0 + movedD3_F)

    # 4. Test restarting from a file
    atoms, calc = restart('calc.gpw', Class=GPAW)
    # Make sure the cached energies and forces are correct
    # without a new calculation
    assert E == pytest.approx(atoms.get_potential_energy())
    assert F == pytest.approx(atoms.get_forces())

    if mode == 'lcao':
        # See issue #1369
        return

    # Make sure the recalculated energies are forces are correct
    atoms.set_positions(atoms.get_positions() + 1e-10)
    assert E == pytest.approx(atoms.get_potential_energy(), abs=1e-5)
    assert F == pytest.approx(atoms.get_forces(), abs=1e-5)

    # 5. Test full blown relaxation.
    from ase.optimize import BFGS
    atoms = get_atoms()
    calc = get_calc(atoms)
    relax = BFGS(atoms)
    relax.run()
    nsteps = relax.nsteps
    #assert atoms.get_distance(0, 1) == pytest.approx(1.8483, abs=1e-2)
    # XXX Replace with a new test
    Egs = atoms.get_potential_energy()
    L = atoms.get_distance(0, 1)

    # 6. Test restarting from a relaxation.
    atoms = get_atoms()
    calc = get_calc(atoms)
    relax = BFGS(atoms, restart='relax_restart')
    for _, _ in zip(relax.irun(), range(3)):
        pass
    calc.write('restart_relax.gpw')
    atoms, calc = restart('restart_relax.gpw', Class=GPAW)
    relax = BFGS(atoms, restart='relax_restart')
    relax.run()

    assert relax.nsteps + 3 == nsteps
    assert atoms.get_distance(0, 1) == pytest.approx(L, abs=1e-2)
    assert atoms.get_potential_energy() == pytest.approx(Egs, abs=1e-4)
