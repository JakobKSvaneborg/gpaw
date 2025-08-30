from myqueue.workflow import run


def workflow():
    with run(script='calculate.py', cores=8, tmax='1h'):
        with run(script='postprocess.py', cores=8):
            run(script='plot.py')
