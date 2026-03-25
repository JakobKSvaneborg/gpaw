from myqueue.workflow import run


def workflow():
    with run(script='gs_MoS2.py', cores=16, tmax='25m'):
        r1 = run(script='bb_MoS2.py', cores=16, tmax='20h')
    with run(script='gs_WSe2.py', cores=16, tmax='25m'):
        r2 = run(script='bb_WSe2.py', cores=16, tmax='20h')
    with r1, r2:
        with run(script='interpolate_bb.py'):
            run(script='interlayer.py')
