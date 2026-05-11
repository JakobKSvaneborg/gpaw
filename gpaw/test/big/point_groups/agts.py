from myqueue.workflow import run


def workflow():
    with run(script='systems.py'):
        run(script='analyse.py', tmax='1h')
