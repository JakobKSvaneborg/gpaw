from myqueue.workflow import run


def workflow():
    with run(script='h2o.py'):
        with run(script='bader.py'):
            run(script='plot.py')
