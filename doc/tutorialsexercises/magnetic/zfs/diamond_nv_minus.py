import json
from ase.parallel import paropen
from ase.build import bulk
from ase.optimize import BFGS
from gpaw import GPAW, PW
from gpaw.zero_field_splitting import zfs


def nv_minus(n: int, relax: bool = False) -> None:
    atoms = bulk('C', 'diamond', cubic=True) * n
    atoms.numbers[0] = 7
    del atoms[1]
    atoms.set_initial_magnetic_moments([1] * len(atoms))
    name = f'NC{8 * n**3 - 2}'
    atoms.calc = GPAW(xc='PBE',
                      mode=PW(ecut=400),
                      charge=-1,
                      parallel={'kpt': 1},
                      txt=name + '.txt')
    atoms.get_forces()
    # atoms.calc.write(name + '.gpw', 'all')
    D1 = zfs(atoms.calc)

    if relax:
        BFGS(atoms,
             logfile=name + '.log').run(fmax=0.05)
        # atoms.calc.write(name + '.relaxed.gpw', 'all')

    D2 = zfs(atoms.calc)
    with paropen(name + '.json', 'w') as fd:
        json.dump(dict(D1=D1.tolist(), D2=D2.tolist()), fd)


if __name__ == '__main__':
    for n in range(2, 4):
        nv_minus(n, relax=True)
