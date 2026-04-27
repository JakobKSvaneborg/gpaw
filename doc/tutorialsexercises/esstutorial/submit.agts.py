from myqueue.workflow import run


def workflow():
    with run(script='gs_mnf2.py', cores=40, tmax='10m'):
        with run(script='mft_allbz.py', cores=40, tmax='1h'):
            run(script='get_Jij.py')
            run(script='get_allJ.py')
