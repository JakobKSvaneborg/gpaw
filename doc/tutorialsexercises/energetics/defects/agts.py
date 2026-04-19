from myqueue.workflow import run


def workflow():
    runs = [run(script='gaas.py', args=[n, charge], cores=cores, tmax='1h')
            for n, cores in [(1, 1), (2, 8), (3, 24), (4, 48)]
            for charge in [0, -3]]
    with run(script='electrostatics.py', cores=24, processes=1, tmax='15m',
             deps=runs):
        run(script='plot_energies.py')
        run(script='plot_potentials.py')
