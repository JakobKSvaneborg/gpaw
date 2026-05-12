from myqueue.workflow import run


def workflow():
    r1 = run(script='dipole.py', cores=4, tmax='1h')
    r2 = run(script='pwdipole.py', cores=4)
    with r1, r2:
        run(script='plot.py')
        run(script='check.py')
