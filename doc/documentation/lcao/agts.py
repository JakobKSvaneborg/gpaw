from myqueue.workflow import run


def workflow():
    run(script='basisgeneration.py')
    run(script='lcao_h2o.py')
