from myqueue.workflow import run


def workflow():
    run(script='h2o_lcao.py')
    r1 = run(script='g2_dm_ui_vs_scf.py', cores=8, tmax='4h')
    with r1:
        run(script='plot_g2.py')
    r2 = run(script='wm_dm_vs_scf.py', cores=8, tmax='1h')
    with r2:
        run(script='plot_h2o.py')
