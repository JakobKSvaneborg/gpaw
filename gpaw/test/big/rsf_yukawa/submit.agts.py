from myqueue.workflow import run

from gpaw import GPAW_NEW


def workflow():
    if GPAW_NEW == 1:
        return
    run(script='lrtddft.py', cores=4)
