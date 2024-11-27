import json
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.data import atomic_numbers
from ase.units import Bohr
from gpaw import GPAW, setup_paths, KohnShamConvergenceError
from gpaw.setup import create_setup
from gpaw.utilities import h2gpts

# Volumes from ACWF [Å^3/atom]:
FCC = [
    0.0,
    2.965, 17.773, 20.224, 7.872, 5.892, 7.322, 7.601, 7.999, 10.147, 24.303,
    37.099, 23.125, 16.495, 14.482, 14.564, 15.881, 21.288, 52.276, 74.004,
    42.194, 24.687, 17.395, 13.905, 11.886, 10.747, 10.260, 10.308, 10.835,
    11.952, 15.162, 18.947, 19.582, 19.318, 20.378, 26.418, 66.042, 91.427,
    54.892, 32.472, 23.213, 18.768, 16.035, 14.513, 13.837, 14.050, 15.325,
    17.839, 22.841, 27.510, 28.009, 27.490, 28.279, 35.105, 87.007, 117.361,
    64.114, 36.947, 26.522, 24.094, 22.765, 22.245, 22.828, 24.992, 27.994,
    30.552, 32.477, 33.892, 34.823, 35.332, 35.704, 28.971, 22.568, 18.839,
    16.458, 15.016, 14.341, 14.505, 15.656, 17.979, 32.348, 31.140, 32.033,
    31.810, 32.563, 39.031, 93.156, 117.163, 71.627, 45.551, 32.184, 25.298,
    21.713, 19.295, 17.802, 17.364, 17.492]
BCC = [
    0.0,
    2.967, 18.030, 20.267, 7.816, 6.139, 6.686, 7.235, 7.786, 10.084, 24.711,
    37.015, 22.917, 16.926, 14.645, 14.230, 15.762, 21.455, 53.355, 73.780,
    42.150, 24.886, 17.267, 13.461, 11.548, 10.781, 10.500, 10.545, 10.895,
    12.005, 15.375, 19.206, 19.269, 19.052, 20.360, 26.784, 67.463, 91.144,
    54.013, 33.030, 22.845, 18.142, 15.793, 14.620, 14.236, 14.474, 15.444,
    17.982, 23.420, 27.781, 27.647, 27.226, 28.515, 35.987, 89.035, 116.842,
    63.305, 37.818, 27.324, 23.141, 21.068, 20.358, 21.646, 26.134, 28.947,
    30.901, 32.289, 33.267, 33.931, 34.358, 34.640, 29.626, 22.305, 18.292,
    16.145, 15.104, 14.781, 15.056, 15.839, 18.042, 29.237, 31.414, 31.970,
    31.635, 32.854, 40.007, 95.447, 116.492, 70.967, 45.944, 32.568, 24.797,
    20.266, 17.808, 16.564, 16.191, 16.521]

# Made with old PAW-potential generator:
old_names = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Na.1',
    'Mg', 'Mg.2', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti',
    'V', 'V.5', 'Cr', 'Mn', 'Mn.7', 'Fe', 'Co', 'Ni', 'Ni.10', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Nb.5',
    'Mo', 'Mo.6', 'Ru', 'Ru.8', 'Rh', 'Rh.9', 'Pd', 'Pd.10', 'Ag', 'Ag.11',
    'Cd', 'In', 'Sn', 'Sb', 'Te', 'Te.16', 'I', 'Xe', 'Cs', 'Ba', 'Hf',
    'Ta', 'Ta.5', 'W', 'W.6', 'Re', 'Os', 'Os.8', 'Ir', 'Ir.9', 'Pt',
    'Pt.10', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Rn']

# Made with new PAW-potential generator:
new_names = [
    'Cr.14',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']


def workflow():
    from myqueue.workflow import run
    setup_paths.append('../potentials')
    name = Path().absolute().name
    for xtal in ['fcc', 'bcc']:
        for mode in ['pw', 'fd', 'lcao']:
            run(function=scan_parameter,
                args=[name, xtal, mode],
                cores=24,
                tmax='1h',
                name=f'{xtal}-{mode}')
    run(function=eggbox_error,
        args=[name],
        cores=24,
        tmax='10h',
        name='eggbox')


def scan_parameter(name: str,
                   xtal: str,
                   mode: str) -> tuple[str, str, str, str,
                                       list[float], list[float]]:
    symbol, _, kind = name.partition('.')
    kind = kind or 'paw'
    Z = atomic_numbers[symbol]

    if xtal == 'fcc':
        vol = 4 * FCC[Z]
    else:
        vol = 2 * BCC[Z]
    atoms = bulk(symbol, xtal, a=vol**(1 / 3))

    params = {'kpts': {'density': 3.0},
              'xc': 'PBE',
              'setups': kind,
              'parallel': {'domain': 1}}

    if mode == 'pw':
        x = list(range(1200, 200, -100))
        scan = [{'mode': {'name': 'pw', 'ecut': ecut}}
                for ecut in x]
    else:
        x = grid_spacings(atoms.cell)
        scan = [{'h': h} for h in x]
        params['mode'] = mode
        if mode == 'lcao':
            params['basis'] = 'dzp'

    results = []
    for i, p in enumerate(scan):
        atoms.calc = GPAW(**params,
                          **p,
                          txt=f'{xtal}-{mode}-{i}.txt')
        try:
            e = atoms.get_potential_energy()
        except KohnShamConvergenceError:
            break
        results.append(e)
    return (symbol, kind, xtal, mode, x, results)


def grid_spacings(cell_cv: np.ndarray) -> list[float]:
    g1 = h2gpts(0.12, cell_cv)[0]
    g2 = h2gpts(0.2, cell_cv)[0]
    g2 = max(min(g2, g1 - 8), 8)
    if g2 == g1:
        g1 += 8
    gs = range(g1, g2 - 4, -4)
    L = (np.linalg.inv(cell_cv)[:, 0]**2).sum()**-0.5
    return [L / g for g in gs]


def eggbox_error(name: str) -> tuple[str, str, list]:
    symbol, _, kind = name.partition('.')
    kind = kind or 'paw'
    Z = atomic_numbers[symbol]
    vol = 4 * FCC[Z]
    atoms = bulk(symbol, 'fcc', a=vol**(1 / 3))

    params = {'kpts': {'density': 3.0},
              'xc': 'PBE',
              'setups': kind,
              'symmetry': 'off',
              'mode': 'fd',
              'parallel': {'domain': 1}}
    results = []
    for h in grid_spacings(atoms.cell):
        atoms.calc = GPAW(**params,
                          h=h,
                          txt=f'eggbox-{h:.2f}.txt')
        try:
            e0 = atoms.get_potential_energy()
            h_v = atoms.calc.density.gd.h_cv[0] * Bohr
            energies = []
            for _ in range(3):
                atoms.positions += h_v / 6
                e = atoms.get_potential_energy()
                energies.append((h, e - e0))
            results.append((h, energies))
        except KohnShamConvergenceError:
            break
    return (symbol, kind, results)


def collect_results(name: str) -> dict:
    symbol, _, kind = name.partition('.')
    kind = kind or 'paw'
    dct = {}
    for xtal in ['fcc', 'bcc']:
        for mode in ['pw', 'fd', 'lcao']:
            dct[f'{xtal}-{mode}'] = json.loads(
                Path(f'{name}/{xtal}-{mode}.result').read_text())
    dct['eggbox'] = json.loads(
        Path(f'{name}/eggbox.result').read_text())

    pot = create_setup(symbol, 'PBE', type=kind)
    nlfer_j = []
    for n, l, f, e, r in zip(pot.n_j,
                             pot.l_j,
                             pot.f_j,
                             pot.data.eps_j,
                             pot.rcut_j):
        nlfer_j.append((n, l, f, e, r))
    dct['nlfer'] = nlfer_j
    dct['nvalence'] = pot.Nv
    return dct


def make_folders() -> None:
    names = old_names + new_names
    for name in names:
        Path(name).mkdir(exist_ok=True)


def collect_all() -> None:
    setup_paths.append('potentials')
    dct = {}
    names = old_names + new_names
    for name in names:
        dct[name] = collect_results(name)
    Path('potentials.json').write_text(json.dumps(dct))


if __name__ == '__main__':
    # make_folders()
    collect_all()
