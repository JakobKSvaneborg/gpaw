from myqueue.workflow import run


def workflow():
    runs = [run(script='PES_CO.py', cores=8, tmax='1h'),
            run(script='PES_H2O.py', cores=8, tmax='1h'),
            run(script='PES_NH3.py', cores=8, tmax='55m')]
    run(script='PES_plot.py', deps=runs)
