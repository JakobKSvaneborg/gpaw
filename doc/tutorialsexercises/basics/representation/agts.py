import pathlib
import numpy as np


def test():
    txt = pathlib.Path('results-400.txt').read_text()
    last_line = txt.split('\n')[-2]
    atomization_energy = float(last_line)
    assert np.isclose(atomization_energy, -11.5211)


def test_ecut():
    txt = pathlib.Path('results-500.txt').read_text()
    last_line = txt.split('\n')[-2]
    atomization_energy = float(last_line)
    assert np.isclose(atomization_energy, -11.6263)


def workflow():
    from myqueue.workflow import run
    with run(script='h2o.py'):
        run(function=test)

    with run(script='h2o_ecut.py'):
        run(function=test_ecut)
