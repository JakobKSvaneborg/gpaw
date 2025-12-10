# creates: benchmark.csv, benchmark.png, systems.csv, score.png, H2-0.xyz
import json
from datetime import date
from pathlib import Path

from ase.geometry.cell import cell_to_cellpar
from gpaw.benchmark.performance_index import PARAMS, REFERENCES
from gpaw.benchmark.systems import systems
from gpaw.calcinfo import get_calculation_info

NAMES = sorted(REFERENCES, key=lambda name: name.split('-')[::-1])


def tables(data) -> None:
    day, score, results = data
    lines = ['name, dt1 [sec], iter1, dt2 [sec], iter2, memory [Gbytes]']
    for name in NAMES:
        if name not in results:
            continue
        e1, t1, i1, m1, e2, t2, i2, m2 = results[name]
        lines.append(
            f'{name:12}, '
            f'{t1:6.1f}, {i1:3}, '
            f'{t2:6.1f}, {i2:3}, '
            f'{m2 * 1e-9:.2f}')
    Path('benchmark.csv').write_text('\n'.join(lines) + '\n')

    lines = [
        'name, cores, IBZ, bands, '
        'a [Å], b [Å], c [Å], '
        'A [°], B [°], C [°]']
    for name in NAMES:
        e, de, cores, t = REFERENCES[name]
        atoms = systems[name]()
        atoms.info.clear()  # remove adsorbate-info (which xyz does not like)
        atoms.write(f'{name}.xyz')
        info = get_calculation_info(atoms, **PARAMS)
        a, b, c, A, B, C = cell_to_cellpar(atoms.cell)
        lines.append(
            f':download:`{name} <{name}.xyz>`, {cores}, '
            f'{len(info.ibz)}, {info.nbands}, '
            f'{a:.1f}, {b:.1f}, {c:.1f}, '
            f'{A:.1f}, {B:.1f}, {C:.1f}')
    Path('systems.csv').write_text('\n'.join(lines) + '\n')


def plot_score(data) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, scores in data.items():
        if mode == 'PW':
            scores = scores[1:]
        X = []
        Y = []
        for day, score in scores:
            X.append(date(*day))
            Y.append(score)
        ax.plot(X, Y, 'o-', label=mode)
    ax.axhline(100.0, ls=':', color='black', label='PW (gpaw-25.7.0)')
    ax.legend()
    ax.set_xlabel('date')
    ax.set_ylabel('score')
    plt.tight_layout()
    plt.savefig('score.png')


def plot(data) -> None:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    titles = [
        '$t_i^0/t_i$',
        'Second step [%]',
        'max_rss [Gbytes]']

    for n, (title, ax) in enumerate(zip(titles, axs)):
        for tag, (day, score, results) in data.items():
            Y = []
            for name in NAMES:
                if name in results:
                    r = results[name]
                    if n == 0:
                        t0 = REFERENCES[name][3]
                        if t0 < 9999999:
                            y = t0 / (r[1] + r[5])
                        else:
                            y = None
                    elif n == 1:
                        y = 100 * r[5] / (r[1] + r[5])
                    else:
                        y = r[7] * 1e-9
                else:
                    y = None
                Y.append(y)
            if n == 2:
                ax.semilogy(Y, 'x-', label=f'{tag} ({day}): {score:.1f}')
            else:
                ax.plot(Y, 'x-', label=f'{tag} ({day}): {score:.1f}')
        ax.set_ylabel(title)
        if n == 0:
            ax.axhline(1.0, ls=':', color='black')
        if n == 2:
            ax.set_xticks(range(len(NAMES)), NAMES, rotation=70, ha='center')
            ax.legend()
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig('benchmark.png')


def main(data):
    plot(data['latest'])
    plot_score(data['scores'])
    tables(data['latest']['PW'])


if 1:  # __name__ == '__main__':
    data = json.loads(Path('benchmarks.json').read_text())
    main(data)
