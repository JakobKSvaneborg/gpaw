from myqueue.workflow import run


def workflow():
    with run(script='gs.py', cores=48, tmax='30m'):
        with run(script='td.py', cores=48, tmax='4h'):
            with run(script='spec.py') as spec:
                run(script='plot_spec.py')

    with run(script='basis.py', folder='mybasis'):
        with run(script='gs.py', cores=48, tmax='30m', folder='mybasis'):
            with run(script='td.py', cores=48, tmax='4h', folder='mybasis'):
                spec_my = run(script='spec.py', folder='mybasis')

    with run(script='gs.py', cores=48, tmax='2h', folder='dzp'):
        with run(script='td.py', cores=40, tmax='5h', folder='dzp'):
            spec_dzp = run(script='spec.py', folder='dzp')

    run(script='plot_spec_basis.py', deps=[spec, spec_my, spec_dzp])
