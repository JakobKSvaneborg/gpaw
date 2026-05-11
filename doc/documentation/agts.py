from myqueue.workflow import run


def workflow():
    run(script='hyperfine_21.py')
    with run(script='cu_calc.py', cores=4, tmax='1h'):
        run(script='cu_plot.py')
        run(function=check)


def check():
    import numpy as np
    from ase.io import read
    energies = []
    k = 20
    for name in ['ITM', 'FD-0.05', 'MV-0.2']:
        e = read(f'Cu-{name}-{k}.txt').get_potential_energy()
        energies.append(e)
    # Extrapolate TM:
    e19 = read('Cu-TM-19.txt').get_potential_energy()
    e20 = read('Cu-TM-20.txt').get_potential_energy()
    e = np.polyval(np.polyfit([20**-2, 19**-2], [e20, e19], 1), 0)
    energies.append(e)
    assert max(energies) - min(energies) < 0.001
