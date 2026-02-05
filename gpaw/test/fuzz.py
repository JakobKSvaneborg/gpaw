from __future__ import annotations

import argparse
import inspect
from functools import cache
from io import StringIO
from random import Random

from ase import Atoms
from ase.build import bulk
from gpaw.dft import DFT
from gpaw.mpi import world


def main(args: str | list[str] = None) -> int:
    if isinstance(args, str):
        args = args.split()

    parser = argparse.ArgumentParser()
    parser.add_argument('--check-serial')
    parser.add_argument('-n', '--runs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_intermixed_args(args)

    seed = args.seed
    n = 0
    while n != args.runs:
        run_random_calculation(seed)
        n += 1
        seed += 1


def log(*args, **kwargs):
    if world.rank == 0:
        print(*args, **kwargs)


def run_random_calculation(seed):
    log('Seed:', seed)
    rng = Random(seed)

    atoms = random_atoms(rng)

    mode = rng.choice(['pw', 'lcao', 'fd'])
    if mode == 'pw':
        mode = {'name': 'pw', 'ecut': rng.uniform(25, 800)}

    kwargs = {'mode': mode}

    if rng.random() < 0.05:
        kwargs['gpts'] = [rng.randint(5, 43) for _ in range(3)]

    if rng.random() < 0.1:
        kwargs['h'] = rng.uniform(0.03, 2.0)

    if rng.random() < 0.8:
        kwargs['kpts'] = {
            'density': rng.uniform(0.1, 4),
            'gamma': rng.random() < 0.5}

    kwargs['parallel'] = rng.choice(parallelizations(world.size))

    out = StringIO()
    log(atoms.symbols.formula)
    log(kwargs)
    try:
        dft = DFT(atoms,
                  **kwargs,
                  txt=out)
        dft.converge(steps=3)
        dft.calculate_energy()
        # dft.calculate_forces()
        # dft.calculate_stress()
    except Exception as ex:
        log(ex)
        assert not isinstance(ex, AssertionError)
        assert isinstance(ex.args[0], str)
        frame = inspect.trace()[-1]
        assert frame  # ...


def completely_random(rnd):
    ...


def random_atoms(rng: Random) -> Atoms:
    if rng.random() < -0.3:
        atoms = completely_random()
    else:
        atoms_candidates = create_atoms_objects()
        atoms = rng.choice(atoms_candidates)

    for c, periodic in enumerate(atoms.pbc):
        if periodic and rng.random() < 0.5:
            atoms = atoms.repeat([1] * c + [2] + (2 - c) * [1])
            if rng.random() < 0.5:
                atoms.pbc[c] = False

    if rng.random() < 0.1:
        magmoms = [rng.uniform(-3, 3) for _ in range(len(atoms))]
        atoms.set_initial_magnetic_moments(magmoms)

    return atoms


@cache
def parallelizations(n: int):
    results = []
    for k in range(1, n + 1):
        if n % k == 0:
            for b in range(1, n // k + 1):
                if n // k % b == 0:
                    results.append({'kpt': k, 'band': b})
    return results


@cache
def create_atoms_objects() -> list[Atoms]:
    candidates = []
    atoms = Atoms('H', magmoms=[1])
    atoms.center(vacuum=2.0)
    candidates.append(atoms)

    atoms = Atoms('H2', [(0, 0, 0), (0, 0.75, 0)])
    atoms.center(vacuum=2.0)
    candidates.append(atoms)

    atoms = bulk('Si', a=5.4)
    candidates.append(atoms)

    atoms = bulk('Fe')
    atoms.set_initial_magnetic_moments([2.3])
    candidates.append(atoms)

    L = 5.0
    atoms = Atoms('Li', cell=[L, L, 1.5], pbc=(0, 0, 1))
    atoms.center()
    candidates.append(atoms)

    return candidates


if __name__ == '__main__':
    raise SystemExit(main())
