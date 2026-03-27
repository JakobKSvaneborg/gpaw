from myqueue.workflow import run


def workflow():
    with run(script='atomize.py', tmax='30m'):
        run(script='relax.py', tmax='30m')
