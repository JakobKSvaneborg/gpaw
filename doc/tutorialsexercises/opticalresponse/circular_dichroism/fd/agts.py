from myqueue.workflow import run


def workflow():
    with run(script='gs.py', cores=24):
        td_jobs = []
        for kick in 'xyz':
            td = run(script='td.py', args=['--', '--kick', kick],
                     cores=24, tmax='6h')
            td_jobs.append(td)
        with run(script='spec.py', deps=td_jobs):
            run(script='plot_spectrum.py')
