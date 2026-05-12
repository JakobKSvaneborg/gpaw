from myqueue.workflow import run


def workflow():
    with run(script='Fe_gs.py', cores=40, tmax='1h'):
        with run(script='Fe_mft.py', cores=40, tmax='2h'):
            run(script='Fe_plot_magnons_vs_rc.py')
            run(script='Fe_plot_magnon_dispersion.py')
    with run(script='Co_gs.py', cores=40, tmax='1h'):
        with run(script='Co_mft.py', cores=40, tmax='6h'):
            run(script='Co_plot_magnons_vs_rc.py')
            run(script='Co_plot_magnon_dispersion.py')
