import json
import sys
from pathlib import Path
from time import time

import numpy as np
from gpaw.benchmark.systems import systems
from gpaw.calcinfo import get_calculation_info
from gpaw.dft import GPAW, PW, MonkhorstPack
from gpaw.mpi import world
from gpaw.utilities.memory import maxrss

params = dict(
    xc='PBE',
    mode=PW(800),
    kpts=MonkhorstPack(density=5.0))


def workflow():
    from myqueue.workflow import run
    for name, function in systems.items():
        if name in {'magic_graphene', 'C6000', 'C2188', 'C676'}:
            continue
        atoms = function()

        # Estimate time:
        info = get_calculation_info(atoms, **params)
        t = (len(info.ibz) * info.nbands * info.ncomponents *
             atoms.cell.volume * 1e-6)
        tmax = '1h'
        if t < 1.0:
            cores = 24
        elif t < 10.0:
            cores = 40
        else:
            cores = 56
            tmax = '3h'
        run(function=work,
            args=[name],
            cores=cores,
            tmax=tmax,
            name=name,
            creates=[f'{name}.json'])


def work(name):
    global params
    extra = Path('params.json')
    if extra.is_file():
        params |= json.loads(extra.read_text())
    atoms = systems[name]()
    # from gpaw import GPAW
    calc = GPAW(
        txt=f'{name}.txt',
        **params)
    atoms.calc = calc

    t1 = time()
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    i1 = calc.dft.scf_loop.niter

    if abs(f1).max() < 0.0001:
        s1 = atoms.get_stress(voigt=False)
        atoms.set_cell(atoms.cell @ (np.eye(3) - 0.02 * s1), scale_atoms=True)
    else:
        atoms.positions += 0.1 * f1
    t1 = time() - t1
    m1 = maxrss()

    t2 = time()
    e2 = atoms.get_potential_energy()
    _ = atoms.get_forces()
    i2 = calc.dft.scf_loop.niter
    t2 = time() - t2
    m2 = maxrss()

    if world.rank == 0:
        Path(f'{name}.json').write_text(json.dumps([e1, t1, i1, m1,
                                                    e2, t2, i2, m2]))


energies = {
    'Bi2Se3': (-21.46195, -0.18655),
    'C60': (-530.92535, -0.44820),
    'diamond': (-18.19611, -0.00000),
    'Ga2N4F4H10': (-99.08900, 0.00013),
    'H2': (-6.77477, 0.11710),
    'LiC8': (-75.37653, 0.66102),
    'magbulk': (-72.37710, -0.00713),
    'metalslab': (-350.06299, -0.01156),
    'MnVS2-slab': (-29.11777, -0.00014),
    'MoS2_tube': (-1291.31046, 7.55276),
    'VI2': (-9.29013, -0.77486),
    'OPt111b': (-153.25143, -1.61599)}


def read(folder: Path, mode: int, eps: float = 0.001) -> dict[str, tuple[float, int]]:
    data = {}
    for name, (e0, de0) in energies.items():
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
            t = 9999.9
            i = 999
        data[name] = (t, i)
    return data


def summary(folders: list[Path], mode: int) -> None:
    alldata = [read(folder, mode) for folder in folders]
    for i, folder in enumerate(folders):
        print(i + 1, folder)
    print('------------' + '+---------------------' * len(folders))
    scores = [0.0] * len(folders)
    N = 0
    for name in energies:
        print(f'{name:10} ', end='')
        times = [data[name][0] for data in alldata]
        iters = [data[name][1] for data in alldata]
        t0 = min(times)
        if t0 == 9999.9:
            print()
            continue
        if max(times) < 9999.9:
            N += 1
        for n, (t, i) in enumerate(zip(times, iters)):
            if t == 9999.9:
                print(' | ------(---) ------%', end='')
            else:
                print(f' | {t:6.1f}({i:3}) {(t / t0 - 1) * 100:+6.1f}%',
                      end='')
            if max(times) < 9999.9:
                scores[n] += t / t0
        print()
    print('------------' + '+---------------------' * len(folders) +
          '\n           ', end='')
    for s in scores:
        print(f'{(s / N - 1) * 100:+21.1f}%', end='')
    print()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=int)
    parser.add_argument('folder', nargs='+')
    args = parser.parse_args()
    summary(
        folders=[Path(folder) for folder in args.folder], mode=args.mode)
