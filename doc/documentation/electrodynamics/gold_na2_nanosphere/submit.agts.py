from myqueue.workflow import run


def workflow():
    with run(script='calculate.py', cores=8, tmax='1h'):
        run(script='plot_geom.py')
        run(script='plot.py')
