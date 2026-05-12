"""Niflheim AGTS crontab-script.

Add this line to crontab on slid2::

  # m h dom mon dow command
  3 3 1,15 * * cd AGTS && ./job.sh

where job.sh is::

  source /etc/bashrc
  module load Python
  python niflheim_agts_crontab.py
"""

import os
import subprocess
from datetime import date
from pathlib import Path

REPO = 'https://gitlab.com/gpaw/gpaw/'


def submit():
    d = date.today()
    root = Path(f'{d.year}-{d.month:02}-{d.day:02}')
    root.mkdir()
    os.chdir(root)
    url = REPO + '-/raw/master/doc/platforms/Linux/Niflheim/gpaw_venv.py'
    subprocess.run(['wget', url])
    subprocess.run(['python3', 'gpaw_venv.py', 'venv'])

    # Make summerschool files available:
    activate = Path('venv/bin/activate')
    activate.write_text(activate.read_text() +
                        'export AGTS_FILES=$HOME/AGTS_FILES/\n')

    # Install qeh from git:
    subprocess.run('source venv/bin/activate && '
                   'pip install -e ../qeh/',
                   shell=True)

    # Submit jobs:
    subprocess.run('source venv/bin/activate && '
                   'mq init && '
                   'mq workflow -p agts.py',
                   shell=True)


if __name__ == '__main__':
    submit()
