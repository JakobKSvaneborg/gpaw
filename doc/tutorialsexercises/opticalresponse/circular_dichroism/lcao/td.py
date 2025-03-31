from gpaw import setup_paths
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.magneticmomentwriter import MagneticMomentWriter

# Insert the path to the created basis set
setup_paths.insert(0, '.')


def main(kick, gauge):
    kick_strength = [0., 0., 0.]
    kick_strength['xyz'.index(kick)] = 1e-5

    td_calc = LCAOTDDFT('gs.gpw', txt=f'td-{kick}.out')

    DipoleMomentWriter(td_calc, f'dm-{gauge}_{kick}.dat')
    MagneticMomentWriter(td_calc, f'mm-{gauge}_{kick}.dat')

    td_calc.absorption_kick(kick_strength,gauge=gauge)
    td_calc.propagate(10, 2000)

    td_calc.write(f'td-{kick}.gpw', mode='all')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kick', default='x')
    parser.add_argument('--gauge', type=str, default='length')
    kwargs = vars(parser.parse_args())
    main(**kwargs)
