from myqueue.workflow import run


def workflow():
    with run(script='MoS2_groundstate.py', cores=8, tmax='30m'):
        with run(script='MoS2_buildingblock.py', cores=8, tmax='2h'):
            with run(script='MoS2_gwqeh.py', cores=4, tmax='1h'):
                run(script='MoS2_analyze.py', cores=1, tmax='5m')
