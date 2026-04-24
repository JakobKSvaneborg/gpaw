from myqueue.workflow import run


def workflow():
    run(script='lrtddft.py', cores=4)
