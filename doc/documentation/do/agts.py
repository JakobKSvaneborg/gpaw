from pathlib import Path

from ase.io import read
from myqueue.workflow import run


def workflow():
    with run(script='constraints.py', cores=8):
        run(function=check_constraints)
    with run(script='si_es.py', cores=8):
        run(function=check_si_es)


def check_constraints():
    text = Path('N-Phenylpyrrole_EX_direct.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Dipole moment:'):
            direct = float(line.split()[-2].replace(')', ''))
    text = Path('N-Phenylpyrrole_EX_from_constrained.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Dipole moment:'):
            constrained = float(line.split()[-2].replace(')', ''))
    assert abs(direct * 4.803 + 3.396) < 0.01
    assert abs(constrained * 4.803 + 10.227) < 0.01


def check_si_es():
    text = Path('si_excited.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Excitation energy Si:'):
            es = float(line.split()[-2])
    assert abs(es - 0.561946) < 1e-4
