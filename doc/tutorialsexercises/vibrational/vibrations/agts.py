from ase.vibrations import Vibrations
from myqueue.workflow import run
from ase.io import read


def workflow():
    with run(script='vibH2O.py', cores=8, tmax='15m'):
        run(function=check)


def check():
    vib = Vibrations(read('h2o.txt'))
    freqs = vib.get_frequencies(method='frederiksen')
    assert abs(freqs[-3:] - [1507, 3635, 3750]).max() < 1


if __name__ == '__main__':
    check()
