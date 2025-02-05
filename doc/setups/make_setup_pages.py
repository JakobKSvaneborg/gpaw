# creates: H.rst, H.png
# ... and all the rest.
import gzip
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_names, atomic_numbers
from ase.units import Hartree
from ase.utils import plural

RST = """\
.. Computer generated reST (make_setup_pages.py)
.. index:: {name}
.. _{name}:

================
{name}
================

PAW-potentials:

.. csv-table::
    :header: name, valence electrons, frozen core electrons

{table}"""

RST1 = """

{electrons}
====================

Radial cutoffs and eigenvalues:

.. csv-table::
    :header: id, occ, eig [eV], cutoff [Bohr]

{table1}

The figure shows convergence of the absolute FCC-energy (blue lines)
and BCC-FCC energy difference (orange lines) relative
to converged numbers (plane-wave calculation at 1200 eV).

.. image:: {dataset}.png

.. csv-table::
    :header: mode, E(FCC) [eV/atom], E(BCC) [eV/atom], E(BCC)-E(FCC) [eV/atom]

{table2}

Egg-box errors in finite-difference mode:

.. csv-table::
    :header: grid-spacing [Å], energy variation [meV]

{table3}"""


def rst(names, data):
    symbol = names[0]
    Z = atomic_numbers[symbol]
    name = atomic_names[Z]

    table = ''
    rst = RST
    for name in names:
        dct = data[name]
        _, _, kind = name.partition('.')
        kind = kind or 'default'

        nv, txt = rst1(dct, name, symbol)
        if kind != 'default':
            kind = f"``'{kind}'``"
        table += f'    {kind},{nv},{Z - nv}\n'
        rst += txt

    with open(symbol + '.rst', 'w') as fd:
        fd.write(rst.format(table=table, name=name))


def rst1(dct, name, symbol):
    table1 = ''
    nv = 0
    for n, l, f, e, rcut in dct['nlfer']:
        n, l, f = (int(x) for x in [n, l, f])
        if n == -1:
            n = ''
        table1 += f"    {n}{'spdf'[l]},{f},{e * Hartree:.3f},"
        if rcut:
            table1 += f'{rcut:.2f}'
            nv += f
        table1 += '\n'

    _, _, eegg = dct['eggbox']
    if eegg:
        table3 = ''
        for h, energies in eegg:
            E = [e for h, e in energies]
            e = np.ptp(E)
            table3 += f'    {h:.2f},{1000 * e:.3f}\n'
    else:
        table3 = '    -,-\n'

    fig = plt.figure(figsize=(8, 5))

    _, _, _, _, xfcc, yfcc = dct['fcc-pw']
    _, _, _, _, xbcc, ybcc = dct['bcc-pw']
    yfcc = np.array(yfcc)
    efcc0 = yfcc[0]
    ebcc0 = ybcc[0]
    table2 = ('    PW(ecut=1200),' +
              f'{efcc0:.3f},{ebcc0:.3f},{ebcc0 - efcc0:.3f}\n')
    n = min(len(yfcc), len(ybcc))
    dy = ybcc[:n] - yfcc[:n]
    dy -= dy[0]
    ax1 = plt.subplot(121)
    ax1.semilogy(xfcc[1:len(yfcc)], abs(yfcc[1:] - yfcc[0]),
                 'C0-', label='PW, E(FCC)')
    ax1.semilogy(xfcc[1:n], abs(dy[1:]), 'C1--', label='PW, E(BCC)-E(FCC)')
    plt.xlabel('Planewave cutoff [eV]')
    plt.ylabel('Error [eV/atom]')
    plt.legend(loc='best')

    ax2 = plt.subplot(122, sharey=ax1)

    for mode, style in [('fd', 's'), ('lcao', 'o')]:
        _, _, _, _, hfcc, efcc = dct[f'fcc-{mode}']
        _, _, _, _, hbcc, ebcc = dct[f'bcc-{mode}']
        efcc = np.array(efcc) - efcc0
        ebcc = np.array(ebcc) - ebcc0
        ax2.semilogy(hfcc[:len(efcc)], abs(efcc), f'C0{style}-',
                     label=f'{mode.upper()}, E(FCC)')
        H = []
        de = []
        for hf, ef in zip(hfcc, efcc):
            _, hb, eb = min((abs(hb - hf), hb, eb)
                            for hb, eb in zip(hbcc, ebcc))
            H.append((hf + hb) / 2)
            de.append(abs(eb - ef))

        ax2.semilogy(H, de, f'C1{style}--',
                     label=f'{mode.upper()}, E(BCC)-E(FCC)')
        if len(efcc) > 0:
            table2 += (
                f'    "{mode.upper()}(h={hfcc[0]:.2f},{hbcc[0]:.2f})",' +
                f'{efcc[0] + efcc0:.3f},' +
                f'{ebcc[0] + ebcc0:.3f},' +
                f'{ebcc[0] + ebcc0 - efcc[0] - efcc0:.3f}\n')

    plt.xlabel('grid-spacing [Å]')
    plt.legend(loc='best')
    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.savefig(name + '.png')
    plt.close(fig)

    nv = dct['nvalence']
    return nv, RST1.format(electrons=plural(nv, 'valence electron'),
                           table1=table1,
                           table2=table2,
                           table3=table3,
                           symbol=symbol,
                           dataset=name)


def main(symbols=None):
    with gzip.open('potentials.json.gz', 'rt') as fd:
        data = json.load(fd)

    for symbol in symbols or data:
        if '.' not in symbol:
            print(symbol, end='')
            sys.stdout.flush()
            rst([symbol] + [name for name in data
                            if name.startswith(symbol + '.')],
                data)
    print()


# if __name__ == '__main__':
if 1:
    # main(['In'])
    main()
