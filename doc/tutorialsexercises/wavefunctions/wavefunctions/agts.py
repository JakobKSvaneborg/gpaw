def workflow():
    from myqueue.workflow import run
    with run(script='CO.py', cores=8, tmax='15m'):
        run(script='../../electronic/dos/dos.py',
            args=['CO.gpw'])
    with run(script='O.py', cores=8, tmax='15m'):
        run(script='O_lumo.py')
