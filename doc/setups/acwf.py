# web-page: acwf.csv
import json
from pathlib import Path
import numpy as np
from ase.data import atomic_numbers


def create_acwf_csv_file():
    table = []
    dct = json.loads(Path('acwf-results.json').read_text())
    for name, data in dct.items():
        Z = atomic_numbers[name.partition('.')[0]]
        row = [Z, name]
        for mode in ['pw', 'lcao']:
            strains = [s for x, s in data[mode]]
            if strains:
                meanae = sum(abs(s) for s in strains) / len(strains) * 100
                maxae = max(abs(s) for s in strains) * 100
            else:
                meanae = np.nan
                maxae = np.nan
            row += [(10 - len(strains)), maxae, meanae]
        table.append(row)

    table.sort(key=lambda row: row[3])
    Path('acwf.csv').write_text(
        ''.join(
            f'{Z},{name},{n1},{mx1:.2f},{mn1:.2f},{n2},{mx2:.2f},{mn2:.2f}\n'
            for Z, name, n1, mx1, mn1, n2, mx2, mn2 in table))


if __name__ == '__main__':
    create_acwf_csv_file()
