from gpaw.utilities.acwf import work


def workflow():
    from myqueue.workflow import run
    run(function=work, args=['FCC', 'Al'],
        cores=8, name='Al-pw')
    run(function=work, args=['FCC', 'Al'], kwargs={'mode': 'lcao'},
        cores=8, name='Al-lcao')
