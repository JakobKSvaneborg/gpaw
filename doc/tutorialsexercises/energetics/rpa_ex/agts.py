import numpy as np


def workflow():
    from myqueue.workflow import run
    with run(script='si_atom_pbe_exx.py', cores=24, tmax='20m'):
        run(function=check_atom)
        with run(script='si_atom_rpa_init_pbe.py', cores=24, tmax='15m'):
            run(script='si_atom_rpa.py',
                cores=24, processes=16, tmax='3h')
    with run(script='si_pbe.py'):
        with run(script='si_pbe_exx.py', cores=4, tmax='15m'):
            run(function=check_si)
    with run(script='si_rpa_init_pbe.py'):
        run(script='si_rpa.py', cores=4, tmax='15m')


def check_si():
    """Test the RPA outputs are consistent and do not change."""
    bulk_results, bulk_benchmark = [], [
        5.421,
        6,
        400.0,
        -10.764252623234455,
        13.86329]

    with open('si_pbe_exx_results.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            x = [float(x) for x in line.strip().split()]
            bulk_results.append(x)

    # make sure results are consistent, the tol here was tested on only one
    # laptop and may to be strict.
    # Loop in case the si.pbe+exx.results.txt file has multiple lines
    for result in bulk_results:
        assert len(result) == 5
        assert np.allclose(result, bulk_benchmark, rtol=1.e-5, atol=1.e-8)


def check_atom():
    refs = [[6.0, -0.6459097136680098, 9.908406700674874],
            [7.0, -0.7741783923740346, 9.805585672225355],
            [8.0, -0.8247705762137586, 9.76898428871515],
            [9.0, -0.8432417920538989, 9.756536401192722],
            [10.0, -0.8498113915010016, 9.75310722624879],
            [11.0, -0.8521763270365798, 9.753051109289967],
            [12.0, -0.8530785423481283, 9.752855601258254]]
    with open('si_atom_pbe_and_exx_energies.txt', 'r') as file:
        results = [
            [float(x) for x in line.split()]
            for line in file.readlines()]

    for (a0, e0, x0), (a, e, x) in zip(refs, results):
        assert a == a0
        assert abs(e - e0) < 0.0001
        assert abs(x - x0) < 0.001
