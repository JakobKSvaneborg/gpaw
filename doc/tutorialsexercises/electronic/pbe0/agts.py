from myqueue.workflow import run


def workflow():
    run(script='gaps.py')
    with run(script='eos.py', cores=4, tmax='10h'):
        run(script='plot_a.py')
