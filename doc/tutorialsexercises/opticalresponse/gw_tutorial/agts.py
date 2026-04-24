import pickle

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

    with run(script='C_converged_mpa.py'):
        run(function=check_mpa)

    with run(script='MoS2_gs_GW.py', tmax='2h'):
        with run(script='MoS2_GWG.py', cores=8, tmax='20m'):
            run(script='MoS2_bs_plot.py')
            run(script='check_gw.py')

    with run(script='C_lcao_groundstate.py', cores=24, tmax='1h'):
        lcao_gw = run(script='C_lcao_gw.py', cores=4, tmax='1h')
    with run(script='C_pw_groundstate.py', cores=24, tmax='1h'):
        pw_gw = run(script='C_pw_gw.py', cores=24, tmax='1h')
    run(script='plot_C_lcao_gw.py', deps=[pw_gw, lcao_gw])


def check_mpa():
    """Check numbers in ReST file."""
    gap_references = [7.19, 7.23]
    for npols in [1, 8]:
        with open(f'C-g0w0_mp{npols}_results_GW.pckl', 'rb') as f:
            results = pickle.load(f)
        gap = results['qp'][0, 0, 1] - results['qp'][0, 0, 0]
        print(npols, gap)
        assert abs(gap - gap_references.pop(0)) < 0.005
