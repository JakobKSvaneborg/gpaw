from myqueue.workflow import run


def workflow():
    with run(script='gs.py', cores=8):
        with run(script='td.py', cores=8, tmax='30m'):
            with run(script='tdc.py', cores=8, tmax='30m'):
                run(script='td_replay.py', cores=8, tmax='30m')
                r1 = run(script='spectrum.py')
                r2 = run(script='td_fdm_replay.py')
        with run(script='ksd_init.py'):
            with run(script='fdm_ind.py', deps=[r2]):
                run(script='ind_plot.py')
            run(script='tcm_plot.py', deps=[r1, r2])
        run(script='spec_plot.py', deps=[r1])
        with run(script='td_pulse.py', cores=8, tmax='30m'):
            run(script='plot_pulse.py')
