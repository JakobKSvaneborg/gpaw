from myqueue.workflow import run


def workflow():
    run(script='nio.py')
    with run(script='n.py'):
        run(script='check.py')
