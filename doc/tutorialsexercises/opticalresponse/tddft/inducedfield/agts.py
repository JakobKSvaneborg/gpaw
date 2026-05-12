from myqueue.workflow import run


def workflow():
    with run(script='timepropagation_calculate.py', cores=8, tmax='1h'):
        with run(script='timepropagation_continue.py', cores=8, tmax='1h'):
            with run(script='timepropagation_postprocess.py', cores=8):
                run(script='timepropagation_plot.py')

    with run(script='casida_calculate.py', cores=8, tmax='1h'):
        with run(script='casida_postprocess.py', cores=8):
            run(script='casida_plot.py')
