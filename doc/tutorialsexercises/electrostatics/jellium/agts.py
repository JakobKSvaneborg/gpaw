from myqueue.workflow import run


def workflow():
    run(script='bulk.py', cores=4)
    with run(script='surface.py', cores=4):
        with run(script='sigma.py'):
            run(script='fig2.py')
