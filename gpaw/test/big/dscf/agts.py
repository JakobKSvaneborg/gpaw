from myqueue.workflow import run


def workflow():
    run(script='dscf.py', cores=8, tmax='13h')
