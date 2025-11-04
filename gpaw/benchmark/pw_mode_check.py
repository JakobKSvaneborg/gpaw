import json
from pathlib import Path
from time import time

import numpy as np
from gpaw.benchmark.systems import systems
from gpaw.dft import GPAW, PW, MonkhorstPack
from gpaw.utilities.memory import maxrss


def workflow():
    from myqueue.workflow import run
    for name, function in systems.items():
        atoms = function()
        if len(atoms) == 2:
            run(function=work, args=[name], cores=24,
                creates=f'{name}.json')


def work(name):
    atoms = systems[name]()
    atoms.calc = GPAW(
        mode=PW(800),
        kpts=MonkhorstPack(density=5.0),
        txt=f'{name}.txt')

    t1 = time()
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
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
    t2 = time() - t2
    m2 = maxrss()
    Path(f'{name}.json').write_text(json.dumps([e1, t1, m1, e2, t2, m2]))


if __name__ == '__main__':
    work('H2')

