from myqueue.workflow import run


def workflow():
    with run(script='molecules.py', tmax='1h'):
        run(script='check.py')
