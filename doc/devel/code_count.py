# creates: lines.png
import datetime as dt
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt


def count(dir, pattern):
    if not os.path.isdir(dir):
        return 0
    files = subprocess.check_output(f'find {dir} -name {pattern}',
                                    shell=True).decode().split()[:-1]
    if not files:
        return 0
    out = subprocess.check_output(f"wc -l {' '.join(files)} | tail -1",
                                  shell=True)
    return int(out.split()[0])


def polygon(x, y1, y2, *args, **kwargs):
    x = np.concatenate((x, x[::-1]))
    y = np.concatenate((y1, y2[::-1]))
    plt.fill(x, y, *args, **kwargs)


def plot_count(dpi=70):
    year, month, f, c, py, test, doc, rst = np.loadtxt('lines.data').T
    date = year + (month - 1) / 12

    fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
    fig.add_subplot(111)
    polygon(date, c + py + test + doc, c + py + test + doc + rst,
            facecolor='m', label='Documentation (.rst)')
    polygon(date, c + py + test, c + py + test + doc,
            facecolor='c', label='Documentation (.py)')
    polygon(date, c + py, c + py + test,
            facecolor='y', label='Tests (.py)')
    polygon(date, c, c + py,
            facecolor='g', label='Python-code (.py) ')
    polygon(date, f, c,
            facecolor='r', label='C-code (.c, .h)')
    polygon(date, f, f,
            facecolor='b', label='Fortran-code')

    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.title('Number of lines')
    plt.savefig('lines.png', dpi=dpi)


def count_lines():
    now = dt.date.today()
    stop = now.year, now.month
    year = 2005
    month = 11

    with open('lines.data', 'w') as fd:
        while (year, month) <= stop:
            hash = subprocess.check_output(
                'git rev-list -n 1 --before="{}-{}-01 12:00" master'
                .format(year, month), shell=True).strip()
            print(year, month, hash)
            subprocess.call(['git', 'checkout', hash])

            c = count('c', r'\*.[ch]')
            py = count('.', r'\*.py')
            test = count('gpaw/test', r'\*.py')
            test += count('test', r'\*.py')
            doc = count('doc', r'\*.py')
            py -= test + doc  # avoid double counting
            rst = count('.', r'\*.rst')
            print(year, month, 0, c, py, test, doc, rst, file=fd)
            month += 1
            if month == 13:
                month = 1
                year += 1


plot_count()
