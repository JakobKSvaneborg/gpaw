# creates: table.csv, table.png
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def main():
    data = defaultdict(dict)
    versions = []
    for path in Path().glob('*.json'):
        version = path.stem
        versions.append(version)
        for d in json.loads(path.read_text()):
            id = (d['longname'], d['processor'], str(d['mpi-ranks']))
            data[id][version] = (d['First step'],
                                 d['Second step'],
                                 d['max_rss'])
    versions.sort()
    rows = [['number', 'name', 'processor', 'cores', 'step 1', 'step 2', 'max rss']]
    for i, id in enumerate(sorted(data)):
        d = data[id]
        row = [str(i)] + list(id)
        t1, t2, m = d['25.1.0']
        row += [f'{t1:.0f}s', f'{t2:.0f}s', f'{m * 1e-9:.1f}GB']
        rows.append(row)
    Path('table.csv').write_text(
        '\n'.join(', '.join(row) for row in rows) + '\n')

    fig, axs = plt.subplots(3, 1, sharex=True)
    for n, (name, ax) in enumerate(zip(['First step',
                                        'Second step',
                                        'max_rss'],
                                       axs)):
        for version in versions:
            Y = []
            for id in sorted(data):
                d = data[id]
                if version in d:
                    y = d[version][n] / d['25.1.0'][n] * 100 - 100
                else:
                    y = None
                Y.append(y)
            ax.plot(Y, 'x-', label=version)
        ax.set_ylabel(name + ' [%]')
        if n == 2:
            ax.legend()
    plt.savefig('table.png')
    # plt.show()


if __name__ == '__main__':
    main()
