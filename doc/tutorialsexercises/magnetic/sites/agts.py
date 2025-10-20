from myqueue.workflow import run


def workflow():
    with run(script='Fe_site_properties.py', cores=40, tmax='5m'):
        run(script='Fe_plot_site_properties.py')
        with run(script='Fe_site_sum_rules.py', cores=40, tmax='20m'):
            run(script='Fe_plot_site_sum_rules.py')
            run(script='test_Fe_site_sum_rules.py')
    with run(script='Co_exchange_parameters.py', cores=40, tmax='1h'):
        run(script='Co_plot_hsp_magnons_vs_rc.py')
        run(script='Co_plot_dispersion.py')
        run(script='test_Co_hsp_magnons.py')
    with run(script='nio_dispersion.py', cores=120, tmax='20h'):
        run(script='plot_nio_dispersion.py')
