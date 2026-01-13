import json
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np
from ase.geometry.cell import cell_to_cellpar

from gpaw.benchmark.systems import systems
from gpaw.calcinfo import get_calculation_info
from gpaw.dft import GPAW
from gpaw.mpi import normalize_communicator
from gpaw.utilities.memory import maxrss

PARAMS = dict(
    xc='PBE',
    mode={'name': 'pw', 'ecut': 800},
    kpts={'density': 5.0},
    setups={'Cr': '14'})


# Reference numbers:
#
# 1) energy
# 2) change in energy after first step
# 3) number of cores
# 4) time in seconds
#
# Initial set of 14 materials for the first bechmark-run
# with old GPAW (version 25.7.0):
REFERENCES0 = {
    'Bi2Se3-3': (-21.46195, -0.18655, 24, 46.74),
    'C60-0': (-530.92535, -0.44820, 24, 204.56),
    'C72-2': (-530.92535, -0.44820, 24, 389.18),
    'C2-3': (-18.19611, -0.00000, 24, 11.11),
    'Ga2F4N4H10-3': (-99.08900, 0.00013, 40, 80.28),
    'H2-0': (-6.77477, 0.11710, 24, 3.78),
    'LiC8-3': (-75.37653, 0.66102, 24, 32.01),
    'Fe8-3M': (-72.37710, -0.00713, 24, 114.55),
    'Al96-2': (-350.06299, -0.01156, 40, 1434.00),
    'Mo60S120-1': (-1291.31046, 7.55276, 56, 6239.00),
    'OPt24-2': (-153.25143, -1.61599, 40, 999.75),
    'CrSi2As4-2M': (-38.89434, -0.17154, 24, 100.10),
    'VI2-2M': (-9.29013, -0.77486, 24, 31.65),
    'Ti2Br6-3': (-32.64699, -0.00286, 24, 155.44)}

RESCALE_FACTOR = 1.0

# New materials for second run
# (new GPAW, master branch Nov. 11 2025):
REFERENCES0 |= {
    'MnVS2-2M': (-29.11777, -0.00014, 24, 98.608),
    'PtLi2O6-2M': (0.0, 0.0, 24, 454.22),
    'V3Cl6-2N': (0.0, 0.0, 24, 3364.039)}

# Score for the 14 systems was 94.34.
# Rescaling to 17 systems:
old = 94.34
new = (old / 100 * 14 + 3) / 17 * 100
RESCALE_FACTOR *= old / new

# New initial magmoms for MnVS2-2M (new GPAW, master branch Nov 25 2025).
# Time for MnVS2-2M system changed from 98.608 to 68.767 seconds:
REFERENCES0['MnVS2-2M'] = (-29.11777, -0.00014, 24, 68.767)

# New stuff not yet included in benchmark:
REFERENCES = REFERENCES0 | {
    'ErGe-2M': (0.0, 0.0, 24, 9999999),
    'Mn2O2-3M': (0.0, 0.0, 24, 9999999),
    'Fe8O8-3M': (0.0, 0.0, 40, 9999999)}


def score(data: dict[str, float]) -> tuple[float, int]:
    """GPAW's PW-index (or score).

    With `N` materials and times for completion of each
    material `t_i`, we get this index (normalized to 100
    for the first run with reference times `t_i^0`):::

                   0
              N   t
       100 α ---   i
       ----- >   ----.
         N   ---  t
             i=1   i

    The rescaling factor α is used for rescaling the
    index when new materials are added or hardware is updated.
    """
    s = 0.0
    n = 0
    for name, (_, _, _, tref) in REFERENCES0.items():
        if name in data:
            s += tref / data[name]
            n += 1
    return 100 * RESCALE_FACTOR * s / len(REFERENCES0), n


def workflow(skip: list[str] | None = None) -> list:
    """MyQueue workflow."""
    from myqueue.workflow import run
    handles = []
    for name, (_, _, cores, _) in REFERENCES.items():
        if skip and name in skip:
            continue
        tmax = '2h'
        if cores == 24:
            nodename = 'xeon24el8'
        if cores == 40:
            nodename = 'xeon40el8_clx'
            tmax = '3h'
        elif cores == 56:
            nodename = 'xeon56'
            tmax = '5h'

        handle = run(function=work,
                     args=[name],
                     cores=cores,
                     tmax=tmax,
                     nodename=nodename,
                     name=name,
                     creates=[f'{name}.json'])
        handles.append(handle)
    return handles


def work(name: str,
         params: dict | None = None,
         world=None) -> None:
    """Do two steps."""
    world = normalize_communicator(world)

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

    # Warmup:
    atoms.calc = GPAW(
        txt=None,
        convergence={'maximum iterations': 3},
        **params)
    atoms.get_potential_energy()

    atoms.calc = GPAW(
        txt=f'{name}.txt',
        **params)

    # First step:
    t1 = time()
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    i1 = atoms.calc.dft.scf_loop.niter

    if name in {'C2-3', 'Fe8-3M', 'Mn2O2-3M'}:
        # These systems have zero forces by symmetry
        assert abs(f1).max() < 0.0001
        if atoms.calc.params.mode.name == 'pw':
            stress = atoms.get_stress(voigt=False)
        else:
            # LCAO and FD-mode does not do stress
            s = {'C2-3': -0.0014,
                 'Fe8-3M': 0.0364,
                 'Mn2O2-3M': 0.0382}[name]
            stress = np.diag([s, s, s])
        atoms.set_cell(atoms.cell @ (np.eye(3) - 0.02 * stress),
                       scale_atoms=True)
    else:
        atoms.positions += 0.1 * f1
    t1 = time() - t1
    m1 = maxrss()

    # Second step:
    t2 = time()
    e2 = atoms.get_potential_energy()
    atoms.get_forces()
    i2 = atoms.calc.dft.scf_loop.niter
    t2 = time() - t2
    m2 = maxrss()

    atoms.calc.__del__()  # make sure we get timing info in log-file

    if world.rank == 0:
        Path(f'{name}.json').write_text(json.dumps([e1, t1, i1, m1,
                                                    e2, t2, i2, m2]))


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
                pass  # print(path, x[0] - e0)
            if abs(x[4] - (e0 + de0)) > eps / 10:
                pass  # print(path, 'D', x[4] - (e0 + de0))
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
            t = np.inf
            i = 999
        data[name] = (t, i)
    return data


def summary(folders: list[Path], mode: int) -> None:
    from gpaw.new.logger import GREEN, RESET
    alldata = [read(folder, mode) for folder in folders]
    for i, folder in enumerate(folders):
        print(i + 1, folder)
    print('-----------------' + '+---------------------' * len(folders))
    scores = [0.0] * len(folders)
    N = 0
    for r, name in enumerate(REFERENCES):
        print(f'{r + 1:2} {name:12} ', end='')
        times = [data[name][0] for data in alldata]
        iters = [data[name][1] for data in alldata]
        t0 = min(times)
        if t0 == np.inf:
            print()
            continue
        if max(times) < np.inf:
            N += 1
        for n, (t, i) in enumerate(zip(times, iters)):
            if t == np.inf:
                line = ' | ------(---) ------%'
            else:
                percent = f'{(t / t0 - 1) * 100:+6.1f}%'
                if t == t0:
                    percent = GREEN + percent + RESET
                line = f' | {t:6.1f}({i:3}) {percent}'
            print(line, end='')
            if max(times) < np.inf:
                scores[n] += t / t0
        print()
    print('-----------------' + '+---------------------' * len(folders) +
          '\n           ', end='')
    for s in scores:
        print(f'{(s / N - 1) * 100:+21.1f}%', end='')
    print('\n           ', end='')
    for data in alldata:
        s, _ = score({name: t for name, (t, i) in data.items()})
        print(f'{s:22.2f}', end='')
    print()


def average(folders: list[Path]) -> None:
    data: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(8))
    for folder in folders:
        for path in folder.glob('*.json'):
            x = json.loads(path.read_text())
            data[path.stem] += np.array(x)
    for name, x in data.items():
        e1, t1, i1, m1, e2, t2, i2, m2 = x / len(folders)
        print(
            f'    {name!r}: ('
            f'{e1:.6f}, {t1:.3f}, {round(i1):.0f}, {int(m1)}, '
            f'{e2:.6f}, {t2:.3f}, {round(i2):.0f}, {int(m2)}),')


def main(arguments: list[str] | None = None):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '-m', '--mode', type=int, default=3,
        help='1: first step, 2: second step, 3: both (default).')
    parser.add_argument(
        'folder', nargs='*',
        help='Folder with <name>.json files.')
    parser.add_argument(
        '-a', '--average', action='store_true',
        help='Write average.')
    args = parser.parse_args(arguments)
    if args.folder:
        folders = [Path(folder) for folder in args.folder]
        summary(folders=folders, mode=args.mode)
        if args.average:
            average(folders)
        return

    print('name       natoms ndim IBZ spin bands cores  vol '
          '(lengths)          (angles)')
    for name, (e, de, cores, t) in REFERENCES.items():
        atoms = systems[name]()
        info = get_calculation_info(atoms, **PARAMS)
        print(f'{name:12} {len(atoms):4}    {atoms.pbc.sum()}',
              end=' ')
        print(f'{len(info.ibz):3}    {info.ncomponents}   {info.nbands:3}',
              end='')
        print(f' {cores} {atoms.cell.volume:7.1f}',
              end=' ')
        a, b, c, A, B, C = cell_to_cellpar(atoms.cell)
        print(f'({a:5.1f},{b:5.1f}{c:5.1f}) ({A:5.1f},{B:5.1f},{C:5.1f})')


if __name__ == '__main__':
    main()
