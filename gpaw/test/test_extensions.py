import pytest
from gpaw.new.input_parameters import register
from ase.units import Hartree, Bohr
import numpy as np


class ExtensionParameter:
    def build(self, atoms):
        raise NotImplementedError


"""
class D3Extension:
    def __init__(self, params, atoms):
        self.params = params
        self.atoms = atoms

    def update_forces_postscf(self, F_av):
        from ase.calculators.dftd3 import PureDFTD3
        atoms = self.atoms.copy()
        atoms.calc = PureDFTD3(xc=self.params.xc, **self.params.kwargs)
        F_av += self.atoms.get_forces()

    def update_energy_postscf(self, energies):
        from ase.calculators.dftd3 import PureDFTD3
        atoms = atoms.copy()
        atoms.calc = PureDFTD3(xc=self.params.xc, **self.params.kwargs)
        energies['D3'] += atoms.get_potential_energy()

@register
class D3(ExtensionParameter):
    def __init__(self, *, xc, **kwargs):
        self.xc = xc
        self.kwargs = kwargs

    def build(self, atoms):
        return D3Extension(self, atoms)
"""


class Extension:
    name = 'unnamed extension'

    def __init__(self, atoms, domain_comm):
        ...

    def get_energy_contribution(self) -> float:
        raise NotImplementedError

    def force_contribution(self):
        raise NotImplementedError

    def move_atoms(self, atoms) -> None:
        raise NotImplementedError


@register
class Spring(ExtensionParameter):
    def __init__(self, *, a1, a2, l, k):
        self.a1, self.a2, self.l, self.k = a1, a2, l, k

    def build(self, atoms, domain_comm):
        class EnergyAdder(Extension):

            @property
            def name(_self):
                return f'Spring k={self.k}'

            def __init__(self, atoms):
                self._calculate(atoms)

            def _calculate(_self, atoms):
                D = atoms.get_distance(self.a1, self.a2) - self.l
                v = atoms.positions[self.a1] - atoms.positions[self.a2]
                v /= np.linalg.norm(v)
                F = self.k * D / Hartree * Bohr
                _self.E = 1 / 2 * self.k * D**2 / Hartree
                _self.F_av = np.zeros((len(atoms), 3))
                if domain_comm.rank == 0:
                    _self.F_av[self.a1, :] = -v * F
                    _self.F_av[self.a2, :] = v * F

            def force_contribution(self):
                return self.F_av

            def get_energy_contribution(self):
                return self.E

            def move_atoms(self, atoms):
                self._calculate(atoms)

        return EnergyAdder(atoms)

    def todict(self):
        return dict(a1=self.a1, a2=self.a2, l=self.l, k=self.k)


@pytest.mark.parametrize('parallel', [(1, 1), (1, 2), (2, 2), (2, 1)])
@pytest.mark.parametrize('mode', [{'name': 'pw', 'ecut': 400}, 'fd', 'lcao'])
def test_extensions(mode, parallel, in_tmp_dir):
    ktot = 20

    from gpaw.new.ase_interface import GPAW
    from gpaw import restart
    from gpaw.mpi import world
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

    atoms = get_atoms()

    def get_calc(atoms):
        # To test multiple extensions, create two sprigs which add
        # up to k=ktot, which is what is tested in this test
        calc = GPAW(extensions=[Spring(a1=0, a2=1, l=2, k=4),
                                Spring(a1=0, a2=1, l=2, k=ktot - 4)],
                    symmetry='off',
                    parallel={'band': band, 'domain': domain},
                    kpts=(2, 1, 1),
                    mode=mode)
        atoms.calc = calc
        return calc

    calc = get_calc(atoms)

    E, F = atoms.get_potential_energy(), atoms.get_forces()

    # Write the GPW file for the restart test later on (4.)
    print('Wrote the potential energy', E)
    calc.write('calc.gpw')

    # 2. Test that moving the atoms works after an SFC convergence
    atoms.positions[0, 2] -= 0.1
    movedE, movedF = atoms.get_potential_energy(), atoms.get_forces()

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
    assert E == pytest.approx(E0 + 1 / 2 * ktot * (l - 2)**2)
    assert F[0, 2] == pytest.approx(F0[0, 2] - ktot * (l - 2))

    # Evaluate the reference energy and forces also for the moved atoms
    atoms.positions[0, 2] -= 0.1
    movedE0, movedF0 = atoms.get_potential_energy(), atoms.get_forces()
    l = atoms.get_distance(0, 1)
    assert movedE == pytest.approx(movedE0 + 1 / 2 * ktot * (l - 2)**2)
    assert movedF[0, 2] == pytest.approx(movedF0[0, 2] - ktot * (l - 2))

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
    assert atoms.get_distance(0, 1) == pytest.approx(1.8483, abs=1e-2)
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
