from myqueue.workflow import run
from gpaw import GPAW_NEW


def workflow():
    if GPAW_NEW:
        return
    run(script='lrtddft.py', cores=4)
