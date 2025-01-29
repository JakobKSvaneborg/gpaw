import pathlib
import numpy as np


def test():
    txt = pathlib.Path('results-0.20.txt').read_text()
    last_line = txt.split('\n')[-2]
    atomization_energy = float(last_line)
    assert np.isclose(atomization_energy, -11.6605)


def workflow():
    from myqueue.workflow import run
    with run(script='h2o.py'):
        run(function=test)
