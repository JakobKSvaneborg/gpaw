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
        13.867018289712249,
    ]
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
    isolated_results, isolated_benchmark = [], [
        [6.0, -0.664402266578, 9.88627820237],
        [7.0, -0.778484948334, 9.79998115788],
        [8.0, -0.82500272946, 9.76744817185],
        [9.0, -0.841856681349, 9.75715732758],
        [10.0, -0.848092042293, 9.75399390142],
        [11.0, -0.850367362642, 9.75296805021],
        [12.0, -0.85109735188, 9.75265131464]
    ]
    with open('si_atom_pbe_and_exx_energies.txt', 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            x = [float(x) for x in line.strip().split()]
            isolated_results.append(x)

    for result, benchmark in zip(isolated_results, isolated_benchmark):
        assert len(result) == 3
        print(result, benchmark)
        assert np.allclose(result, benchmark, rtol=1.e-5, atol=1.e-8)
