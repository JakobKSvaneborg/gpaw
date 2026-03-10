"""Niflheim benchmark crontab-script.

Add this line to crontab on slid2::

  # m h dom mon dow command
  3 3 1,15 * * cd BENCHMARKS && ./job.sh

where job.sh is::

  source /etc/bashrc
  module load Python
  python niflheim_crontab.py
"""

import subprocess
from pathlib import Path

REPO = 'https://gitlab.com/gpaw/gpaw/'


def submit():
    url = REPO + '-/raw/master/doc/platforms/Linux/Niflheim/gpaw_venv.py'
    subprocess.run(['wget', url])
    venv = 'venv-benchmark'
    subprocess.run(['python3', 'gpaw_venv.py', venv])

    for mode in ['pw', 'lcao', 'fd']:
        Path(mode).mkdir()
    Path('lcao/params.json').write_text(
        '{"mode": "lcao", "basis": "dzp", "h": 0.15}\n')
    Path('fd/params.json').write_text(
        '{"mode": "fd", "h": 0.15}\n')
    wf = f'{venv}/gpaw/gpaw/benchmark/performance_index.py'
    subprocess.run(f'source {venv}/bin/activate && '
                   'mq init && '
                   f'mq workflow {wf} pw lcao fd',
                   shell=True)


if __name__ == '__main__':
    submit()
