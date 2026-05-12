from myqueue.workflow import run


def workflow():
    with run(script='CO.py'):
        run(script='CO2cube.py')
