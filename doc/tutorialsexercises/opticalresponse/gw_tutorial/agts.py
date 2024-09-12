from myqueue.workflow import run


def workflow():
    with run(script='C_ecut_k_conv_GW.py', cores=24, tmax='10h') as r:
        run(script='C_ecut_k_conv_plot_GW.py')
        run(script='C_ecut_extrap.py')
        with run(script='C_ecut_automatic_extrapolate.py',
                 cores=24, tmax='3h'):
            run(script='C_ecut_automatic_extrapolate_plot.py')

    with run(script='C_frequency_conv.py', tmax='30h'):
        with run(script='C_frequency_conv_plot.py'):
            run(script='C_equal_test.py', deps=[r])

    with run(script='MoS2_gs_GW.py', tmax='2h'):
        with run(script='MoS2_GWG.py', cores=8, tmax='20m'):
            run(script='MoS2_bs_plot.py')
            run(script='check_gw.py')
