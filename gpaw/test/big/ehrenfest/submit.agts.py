from myqueue.workflow import run
from gpaw import GPAW_NEW


def workflow():
    run(script='na2_md.py', cores=8, tmax='2h')
    if GPAW_NEW:
        return
    run(script='na2_osc.py', cores=8, tmax='40h')
    run(script='h2_osc.py', cores=8, tmax='2h')
    run(script='n2_osc.py', cores=40, tmax='15h')
