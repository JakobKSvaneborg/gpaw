from myqueue.workflow import run


def workflow():
    with run(script='al111.py'):
        run(script='stm.py')
