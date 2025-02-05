def workflow():
    from myqueue.workflow import run
    with run(script='Al_fcc.py', cores=2, tmax='15m'):
        run(script='../../electronic/dos/dos.py',
            args=['Al-fcc.gpw'])
    run(script='Al_bcc.py', cores=2, tmax='15m')
    run(script='Al_fcc_vs_bcc.py', cores=2)
    run(script='Al_fcc_modified.py', cores=2)
