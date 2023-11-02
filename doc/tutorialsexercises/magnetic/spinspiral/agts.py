import numpy as np
from myqueue.workflow import run


def workflow():
    with run(script='nii2_sgs.py', cores=40, tmax='2h'):
        run(script='plot.py')
        run(function=check)

        with run(script='nii2_soc.py', cores=16, tmax='30m'):
            run(script='plot_soc.py')
            run(function=check_soc)


def check():
    data = np.load('data.npz')
    energies = data['energies']
    energies = (energies - energies[0]) * 1000
    magmoms = data['magmoms']
    print(energies, magmoms)

    assert np.argmin(energies) == 24
    assert abs(max(energies) - min(energies) - 67.82) < 0.01
    assert abs(magmoms[0] - 1.81) < 0.01


def check_soc():
    data = np.load('soc_data.npz')
    soc = data['soc']
    soc = (soc - soc[0]) * 1000
    print(soc)

    assert np.argmin(soc) == 3394
    assert abs(max(soc) - min(soc) - 3.44) < 0.01


if __name__ == '__main__':
    check()
    check_soc()
