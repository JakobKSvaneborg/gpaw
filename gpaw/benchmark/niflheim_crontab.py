"""Niflheim benchmark crontab-script.

Add this line to crontab on slid2::

  # m h dom mon dow command
  3 3 1,15 * * cd BENCHMARKS && ./job.sh

where job.sh is::

  source /etc/bashrc
  module load Python
  python niflheim_crontab.py
"""

import os
import subprocess
from datetime import date
from pathlib import Path
import json
import sys

REPO = 'https://gitlab.com/gpaw/gpaw/'


def submit():
    d = date.today()
    root = Path(f'{d.year}-{d.month:02}-{d.day:02}')
    root.mkdir()
    os.chdir(root)
    url = REPO + '-/raw/master/doc/platforms/Linux/Niflheim/gpaw_venv.py'
    subprocess.run(['wget', url])
    subprocess.run(['python3', 'gpaw_venv.py', 'venv'])

    for mode in ['pw', 'lcao', 'fd']:
        Path(mode).mkdir()
    Path('lcao/params.json').write_text(
        '{"mode": "lcao", "basis": "dzp", "h": 0.15}\n')
    Path('fd/params.json').write_text(
        '{"mode": "fd", "h": 0.15}\n')
    wf = 'venv/gpaw/gpaw/benchmark/performance_index.py'
    subprocess.run(f'source venv/bin/activate && '
                   'mq init && '
                   f'mq workflow {wf} pw lcao fd',
                   shell=True)


def update(root: Path):
    from gpaw.benchmark.performance_index import read, score
    file = Path(
        'gpaw-web-page-data/gpaw_web_page_data/benchmarks/benchmarks.json')
    data = json.loads(file.read_text())
    date = [int(x) for x in root.name.split('-')]
    for mode in ['pw', 'lcao', 'fd']:
        dct = read(root / mode, 3)
        s, n = score({name: t for name, (t, i) in dct.items()})
        s = round(s, 2)
        print(mode, s, n)
        data['scores'][mode.upper()].append([date, s])
    file.write_text(json.dumps(data, indent=2))
    for path in root.glob('*/*.out'):
        path.unlink()
    for path in root.glob('*/*.err'):
        if not path.read_bytes():
            path.unlink()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        submit()
    else:
        update(Path(sys.argv[1]))
