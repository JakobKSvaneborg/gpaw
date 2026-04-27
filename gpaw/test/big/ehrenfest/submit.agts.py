from myqueue.workflow import run


def workflow():
    run(script='na2_md.py', cores=8, tmax='2h')
    run(script='na2_osc.py', cores=8, tmax='40h')
    run(script='h2_osc.py', cores=8, tmax='2h')
    run(script='n2_osc.py', cores=40, tmax='15h')
