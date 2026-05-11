from myqueue.workflow import run


def workflow():
    with run(script='Pt_gs.py', cores=4, tmax='20m'):
        with run(script='Pt_bands.py', cores=24, tmax='2h'):
            run(script='plot_Pt_bands.py')

    with run(script='WS2_gs.py', cores=4, tmax='20h'):
        with run(script='WS2_bands.py', cores=24):
            run(script='plot_WS2_bands.py')

    with run(script='Fe_gs.py', cores=4, tmax='20m'):
        with run(script='Fe_bands.py', cores=24):
            run(script='plot_Fe_bands.py')

    with run(script='gs_Bi2Se3.py', cores=4, tmax='2h'):
        with run(script='Bi2Se3_bands.py', cores=48, tmax='5h'):
            run(script='plot_Bi2Se3_bands.py', tmax='2h')
        with run(script='high_sym.py', cores=4, tmax='30h'):
            run(script='parity.py', tmax='5h')

    with run(script='gs_Co.py', cores=48, tmax='2h'):
        with run(script='anisotropy.py', tmax='5h'):
            run(script='plot_anisotropy.py')

    run(script='NiBr2_anisotropy.py', cores=12, tmax='2h')
