from myqueue.workflow import run


def workflow():
    runs = [run(script='ruslab.py', cores=8, tmax='10h'),
            run(script='ruslab.py', args=['H'], cores=8, tmax='10h'),
            run(script='ruslab.py', args=['N'], cores=8, tmax='10h'),
            run(script='ruslab.py', args=['O'], cores=16, tmax='15h'),
            run(script='molecules.py', cores=8, tmax='20m')]
    run(script='results.py', deps=runs)
