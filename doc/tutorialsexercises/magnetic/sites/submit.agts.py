from myqueue.workflow import run


def workflow():
    with run(script='Fe_site_properties.py', cores=40, tmax='5m'):
        run(script='Fe_plot_site_properties.py')
        with run(script='Fe_site_sum_rules.py', cores=40, tmax='20m'):
            run(script='Fe_plot_site_sum_rules.py')
            run(script='test_site_sum_rules.py')
