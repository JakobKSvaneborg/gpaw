from myqueue.workflow import run

from gpaw import GPAW_NEW


def workflow():
    run(script='benzene-dimer-T-shaped.py', cores=96, tmax='20h')
    if GPAW_NEW:
        return
    run(script='adenine-thymine_complex_stack.py', cores=4, tmax='2h')
    run(script='graphene_hirshfeld.py')
