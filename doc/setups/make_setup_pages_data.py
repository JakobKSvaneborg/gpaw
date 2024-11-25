from collections import defaultdict

import ase.db
from ase.io.jsonio import encode

from gpaw.atom.check import check, summary, all_names
from gpaw.setup import create_setup

con = ase.db.connect('datasets.db')

all_names[:] = [
    # 'Li', 'Cr.14',
    # 'La', 'Ce',
    'Pr',
    'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']


def run():
    for name in all_names:
        check(con, name, lcao=False)

    data = {}
    for name in all_names:
        data = defaultdict(list)
        symbol, type, nlfer_j = nlfer(name)
        energies = summary(con, name)
        data[symbol].append((type, nlfer_j, energies))

    with open('datasets.json', 'w') as fd:
        fd.write(encode(data))


def nlfer(name: str):
    if '.' in name:
        symbol, kind = name.split('.')
    else:
        symbol = name
        kind = 'paw'

    pot = create_setup(symbol, 'PBE', type=kind)
    print(dir(pot))
    nlfer_j = []
    for n, l, f, e, r in zip(pot.n_j,
                             pot.l_j,
                             pot.f_j,
                             pot.data.eps_j,
                             pot.rcut_j):
        nlfer_j.append((n, l, f, e, r))
    return symbol, kind, nlfer_j


if __name__ == '__main__':
    nlfer('Cr.14')
