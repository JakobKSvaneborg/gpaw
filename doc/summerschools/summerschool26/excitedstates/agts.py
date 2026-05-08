from myqueue.workflow import run


def workflow():
    with run(script='es1.py', tmax='13h'):
        with run(script='es2.py', tmax='13h', cores=8):
            with run(script='es3.py', tmax='13h', cores=8):
                run(script='es4.py', tmax='13h', cores=8)
