from myqueue.workflow import run


def workflow():
    with run(script='gs_Si.py', cores=4, tmax='20m'):
        run(script='bse_plus_Si.py', cores=40, tmax='1h')

    with run(script='gs_MoS2.py', cores=4, tmax='20m'):
        run(script='bse_plus_MoS2.py', cores=40, tmax='1h')
