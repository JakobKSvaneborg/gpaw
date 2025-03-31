from myqueue.workflow import run


def workflow():
    spec_jobs = []

    # Do aug.dzp calculations
    with run(script='basis.py'):
        with run(script='gs.py', cores=4):
            # Simple calculation
            td_jobs = []
            for kick in 'xyz':
                for gauge in ['length', 'velocity']:
                    td = run(script='td.py',
                             args=['--', '--kick', kick, '--gauge', gauge],
                             cores=4, tmax='1h')
                    td_jobs.append(td)
            spec = run(script='spec.py', deps=td_jobs)
            spec_jobs.append(spec)

            # Different origins
            td_jobs = []
            for kick in 'xyz':
                for gauge in ['length', 'velocity']:
                    td = run(script='td_origins.py',
                             args=['--', '--kick', kick, '--gauge', gauge],
                             cores=4, tmax='1h')
                    td_jobs.append(td)
            spec = run(script='spec_origins.py', deps=td_jobs)
            spec_jobs.append(spec)

    # Do dzp calculations
    with run(script='gs.py', cores=4, folder='dzp'):
        td_jobs = []
        for kick in 'xyz':
            td = run(script='td.py', args=['--', '--kick', kick],
                     cores=4, tmax='1h', folder='dzp')
            td_jobs.append(td)
        spec = run(script='spec.py', deps=td_jobs, folder='dzp')
        spec_jobs.append(spec)

    # Plot spectra
    run(script='plot_spectra.py', deps=spec_jobs)
    run(script='plot_spectra_origins.py', deps=spec_jobs)
