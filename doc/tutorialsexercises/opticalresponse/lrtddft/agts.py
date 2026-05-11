from myqueue.workflow import run


def workflow():
    with run(script='Na2TDDFT.py', cores=2, tmax='1h'):
        run(script='part2.py')
    with run(script='ground_state.py', cores=8):
        run(script='spectrum.py')
