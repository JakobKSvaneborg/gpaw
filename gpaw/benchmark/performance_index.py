import json
from pathlib import Path
from time import time

import numpy as np
from ase.geometry.cell import cell_to_cellpar
from gpaw import GPAW, GPAW_NEW
from gpaw.benchmark.systems import systems
from gpaw.calcinfo import get_calculation_info
from gpaw.mpi import world
from gpaw.utilities.memory import maxrss

PARAMS = dict(
    xc='PBE',
    mode={'name': 'pw', 'ecut': 800},
    kpts={'density': 5.0})


def workflow():
    """MyQueue workflow."""
    from myqueue.workflow import run
    for name, (_, _, cores, _) in REFERENCES.items():
        tmax = '1h'
        nodename = None
        if cores == 40:
            nodename = 'xeon40el8_clx'
        elif cores == 56:
            tmax = '3h'

        run(function=work,
            args=[name],
            cores=cores,
            tmax=tmax,
            nodename=nodename,
            name=name,
            creates=[f'{name}.json'])


def work(name: str, params: dict | None = None) -> None:
    """Do two steps."""

    params = params or PARAMS.copy()
    extra = Path('params.json')
    if extra.is_file():
        params |= json.loads(extra.read_text())

    atoms = systems[name]()

    # Do a non-colinear calculation?
    if hasattr(atoms, '_magmoms'):
        params |= dict(
            magmoms=atoms._magmoms,
            symmetry='off',
            xc='LDA')
        if not GPAW_NEW:
            params['experimental'] = {'magmoms': atoms._magmoms}
            params['parallel'] = {'kpt': 24}
            del params['magmoms']

    calc = GPAW(
        txt=f'{name}.txt',
        **params)
    atoms.calc = calc

    # First step:
    t1 = time()
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    i1 = get_number_of_iterations(calc)

    if abs(f1).max() < 0.0001:
        s1 = atoms.get_stress(voigt=False)
        atoms.set_cell(atoms.cell @ (np.eye(3) - 0.02 * s1), scale_atoms=True)
    else:
        atoms.positions += 0.1 * f1
    t1 = time() - t1
    m1 = maxrss()

    # Second step:
    t2 = time()
    e2 = atoms.get_potential_energy()
    _ = atoms.get_forces()
    i2 = get_number_of_iterations(calc)
    t2 = time() - t2
    m2 = maxrss()

    if world.rank == 0:
        Path(f'{name}.json').write_text(json.dumps([e1, t1, i1, m1,
                                                    e2, t2, i2, m2]))


def get_number_of_iterations(calc) -> int:
    if calc.old:
        return calc.scf.niter
    return calc.dft.scf_loop.niter


# Reference numbers:
#
# 1) energy
# 2) change in energy after first step
# 3) time in seconds
#
REFERENCES = {
    'Bi2Se3': (-21.46195, -0.18655, 24, 55),
    'C60': (-530.92535, -0.44820, 24, 190),
    'C72': (-530.92535, -0.44820, 24, 190),
    'diamond': (-18.19611, -0.00000, 24, 16),
    'Ga2N4F4H10': (-99.08900, 0.00013, 40, 120),
    'H2': (-6.77477, 0.11710, 24, 10.0),
    'LiC8': (-75.37653, 0.66102, 24, 38),
    'magbulk': (-72.37710, -0.00713, 24, 119),
    'metalslab': (-350.06299, -0.01156, 40, 37 * 60),
    'MnVS2-slab': (-29.11777, -0.00014, 24, 3600),
    'MoS2_tube': (-1291.31046, 7.55276, 56, 3700),
    'VI2': (-9.29013, -0.77486, 24, 3090),
    'OPt111b': (-153.25143, -1.61599, 40, 1200),
    'PtO3Li2O3': (0.0, 0.0, 24, 2500),
    'ErGe': (0.0, 0.0, 24, 2500),
    'As4CrSi2': (0.0, 0.0, 24, 103),
    'V3Cl6': (0.0, 0.0, 24, 333),
    'Mn2O2': (0.0, 0.0, 24, 1000),
    'Ti2Br6': (0.0, 0.0, 24, 1000),
    'Fe8O8': (0.0, 0.0, 40, 1000)}

# REFERENCES[...] = (...)


def read(folder: Path,
         mode: int,
         eps: float = 0.001) -> dict[str, tuple[float, int]]:
    """Read <name>.json files."""
    data = {}
    for name, (e0, de0, _, _) in REFERENCES.items():
        path = folder / f'{name}.json'
        if path.is_file():
            x = json.loads(path.read_text())
            if abs(x[0] - e0) > eps:
                print(path, x[0] - e0)
            if abs(x[4] - (e0 + de0)) > eps / 10:
                print(path, 'D', x[4] - (e0 + de0))
            if mode == 1:
                t = x[1]
                i = x[2]
            elif mode == 2:
                t = x[5]
                i = x[6]
            else:
                t = x[1] + x[5]
                i = x[2] + x[6]
        else:
            t = 99999.9
            i = 999
        data[name] = (t, i)
    return data


def summary(folders: list[Path], mode: int) -> None:
    from gpaw.new.logger import GREEN, RESET
    alldata = [read(folder, mode) for folder in folders]
    for i, folder in enumerate(folders):
        print(i + 1, folder)
    print('------------' + '+---------------------' * len(folders))
    scores = [0.0] * len(folders)
    N = 0
    for name in REFERENCES:
        print(f'{name:10} ', end='')
        times = [data[name][0] for data in alldata]
        iters = [data[name][1] for data in alldata]
        t0 = min(times)
        if t0 == 99999.9:
            print()
            continue
        if max(times) < 99999.9:
            N += 1
        for n, (t, i) in enumerate(zip(times, iters)):
            if t == 99999.9:
                line = ' | ------(---) ------%'
            else:
                percent = f'{(t / t0 - 1) * 100:+6.1f}%'
                if t == t0:
                    percent = GREEN + percent + RESET
                line = f' | {t:6.1f}({i:3}) {percent}'
            print(line, end='')
            if max(times) < 99999.9:
                scores[n] += t / t0
        print()
    print('------------' + '+---------------------' * len(folders) +
          '\n           ', end='')
    for s in scores:
        print(f'{(s / N - 1) * 100:+21.1f}%', end='')
    print('\n           ', end='')
    for data in alldata:
        s, _ = score({name: t for name, (t, i) in data.items()})
        print(f'{s:22.2f}', end='')
    print()


def score(data: dict[str, float]) -> tuple[float, int]:
    s = 0.0
    n = 0
    for name, (_, _, _, tref) in REFERENCES.items():
        if name in data:
            s += data[name] / tref
            n += 1
    return 100 / s * n, n


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '-m', '--mode', type=int, default=3,
        help='1: first step, 2: second step, 3: both (default).')
    parser.add_argument(
        'folder', nargs='*',
        help='Folder with <name>.json files.')
    args = parser.parse_args()
    if args.folder:
        summary(folders=[Path(folder) for folder in args.folder],
                mode=args.mode)
        return

    print('name     natoms formula    IBZ spin bands     vol '
          '(lengths)          (angles)')
    for name, (e, de, core, t) in REFERENCES.items():
        atoms = systems[name]()
        info = get_calculation_info(atoms, **PARAMS)
        f = f'{atoms.symbols.formula:ab2}'
        print(f'{name:10} {len(atoms):4} {f:10}', end=' ')
        print(f'{len(info.ibz):3}    {info.ncomponents}   {info.nbands:3}',
              end='')
        print(f' {atoms.cell.volume:7.1f}', end=' ')
        a, b, c, A, B, C = cell_to_cellpar(atoms.cell)
        print(f'({a:5.1f},{b:5.1f}{c:5.1f}) ({A:5.1f},{B:5.1f},{C:5.1f})')


if __name__ == '__main__':
    main()
