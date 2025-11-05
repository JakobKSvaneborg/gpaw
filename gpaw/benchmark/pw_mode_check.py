import json
from pathlib import Path
from time import time

import numpy as np

from gpaw.benchmark.systems import systems
from gpaw.dft import GPAW, PW, MonkhorstPack
from gpaw.utilities.memory import maxrss
from gpaw.mpi import world
from gpaw.calcinfo import get_calculation_info

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
        if t < 1.0:
            cores = 24
        if t < 10.0:
            cores = 40
        else:
            cores = 56
        run(function=work,
            args=[name],
            cores=cores,
            tmax='1h',
            name=name,
            creates=[f'{name}.json'])


def work(name: str) -> None:
    atoms = systems[name]()
    calc = GPAW(
        **params,
        txt=f'{name}.txt')
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


if __name__ == '__main__':
    work('H2')
