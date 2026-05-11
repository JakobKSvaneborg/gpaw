from myqueue.workflow import run


def workflow():
    with run(script='g21gpaw.py', tmax='20h'):
        run(script='analyse.py')
