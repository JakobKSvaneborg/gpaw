from myqueue.workflow import run


def workflow():
    with run(script='al.py', cores=8, tmax='12h'):
        run(script='al_analysis.py')
