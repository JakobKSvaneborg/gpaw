from myqueue.workflow import run


def workflow():
    run(script='bandstructure.py')
    run(script='soc.py')
    with run(script='hse06.py'):
        run(script='plot_hse06.py')
