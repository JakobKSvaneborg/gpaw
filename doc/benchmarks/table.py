# creates: table.csv, table.png
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt


data = defaultdict(dict)
versions = []
for path in Path().glob('*.json'):
    for age in ['old', 'new']:
        version = f'{path.stem}-{age}'
        versions.append(version)
        for d in json.loads(path.read_text()):
            if d['calcinfo'] == age:
                cpus = str(d['mpi-ranks'])
                if d['mpi-ranks'] <= 4:
                    cpus += 'G'
                id = (d['shortname'],
                      d['longname'],
                      d['processor'],
                      cpus)
                data[id][version] = (d['First step'],
                                     d['Second step'],
                                     d['max_rss'])
versions.sort()
rows = [['number',
         'name', 'processor', 'cores',
         'step 1 [s]', 'step 2 [s]', 'max rss [GB]']]
for i, id in enumerate(sorted(data)):
    d = data[id]
    row = [str(i)] + list(id[1:])
    t1, t2, m = d['25.1.0-new']
    row += [f'{t1:.0f}', f'{t2:.0f}', f'{m * 1e-9:.1f}']
    rows.append(row)
Path('table.csv').write_text(
    '\n'.join(', '.join(row) for row in rows) + '\n')

X = [id[0] + ((' ' + id[3]) if id[3].endswith('G') else '')
     for id in sorted(data)]
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
for n, (name, ax) in enumerate(zip(['First step',
                                    'Second step',
                                    'max_rss'],
                                   axs)):
    for version in versions:
        Y = []
        for id in sorted(data):
            d = data[id]
            if version in d:
                y = d[version][n] / d['25.1.0-new'][n] * 100 - 100
            else:
                y = None
            Y.append(y)
        ax.plot(Y, 'x-', label=version)
    ax.set_ylabel(name + ' [%]')
    if n == 2:
        ax.set_xticks(range(len(X)), X, rotation=60, ha='right')
        ax.legend()

plt.savefig('table.png')
# plt.show()
