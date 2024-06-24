from math import cos, pi, sin

from gpaw.new.ase_interface import GPAW
from myqueue.workflow import run


def workflow():
    with run(script='VCl2.py', cores=4):
        run(script='plot.py')
        run(function=check)


def check():
    calc = GPAW('VCl2_gs.gpw')
    M_v, M_av = calc.dft.state.density.calculate_magnetic_moments()
    print(M_v)
    print(M_av)
    m = 2.405
    for x, m_v in zip([0, 2 * pi / 3, 4 * pi / 3], M_av):
        assert abs(m_v - [m * cos(x), m * sin(x), 0]).max() < 0.01


if __name__ == '__main__':
    check()
