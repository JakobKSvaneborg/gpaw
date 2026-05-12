from myqueue.workflow import run


def workflow():
    with run(script='gs_3x3_defect.py', cores=16):
        with run(script='unfold_3x3_defect.py', cores=16):
            run(script='plot_sf.py')
