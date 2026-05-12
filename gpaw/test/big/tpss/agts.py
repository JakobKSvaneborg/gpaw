from myqueue.workflow import run


def workflow():
    run(script='tpss.py', cores=8, tmax='1h')
