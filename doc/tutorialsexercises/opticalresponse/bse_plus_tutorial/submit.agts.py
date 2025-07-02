from myqueue.workflow import run


def workflow():
    with run(script='gs_TiO2.py', cores=80, tmax='2h'):
        with run(script='bse_plus_TiO2.py', cores=80, tmax='4h'):
            run(script='plot_TiO2.py', cores=1, tmax='1h')
            run(script='test_TiO2.py')

    with run(script='gs_MoS2.py', cores=80, tmax='2h'):
        with run(script='bse_plus_MoS2.py', cores=80, tmax='4h'):
            run(script='plot_MoS2.py', cores=1, tmax='1h')
            run(script='test_MoS2.py')
