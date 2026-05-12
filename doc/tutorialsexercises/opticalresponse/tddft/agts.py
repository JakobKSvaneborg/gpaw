from myqueue.workflow import run


def workflow():
    run(script='lcaotddft.py', cores=4, tmax='40m')
    with run(script='Be_gs_8bands.py', cores=2, tmax='20m'):
        run(script='Be_8bands_lrtddft.py', cores=2, tmax='20m')
        run(script='Be_8bands_lrtddft_dE.py', cores=2, tmax='20m')
    run(script='Na2_relax_excited.py', cores=4, tmax='8h')
