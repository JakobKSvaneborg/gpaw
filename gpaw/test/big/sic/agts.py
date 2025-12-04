def workflow():
    from myqueue.workflow import run

    from gpaw import GPAW_NEW
    if GPAW_NEW:
        return
    run(script='scfsic_n2.py', cores=8)
