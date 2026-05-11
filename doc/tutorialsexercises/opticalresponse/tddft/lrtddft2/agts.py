from myqueue.workflow import run


def workflow():
    with run(script='gs.py', cores=8):
        with run(script='unocc.py', cores=8, tmax='30m'):
            with run(script='lr2.py', cores=8):
                with run(script='lr2_restart.py', cores=8):
                    with run(script='lr2_analyze.py', cores=1):
                        run(script='plot_spec.py')
                        run(script='check_consistency_with_docs.py')
