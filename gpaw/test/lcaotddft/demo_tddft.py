import numpy as np

from ase import Atoms

from gpaw import GPAW
from gpaw.tddft import TDDFT
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.new.rttddft import RTTDDFT
from gpaw.tddft.units import as_to_au, autime_to_asetime


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='Demo script for new implementation of RT-TDDFT')
    parser.add_argument('--all', action='store_true', help='Run everything.')
    parser.add_argument('--lcao-gs', action='store_true',
                        help='Run old ground state LCAO calculation. '
                        'Creates lcao_gs.gpw.')
    parser.add_argument('--fd-gs', action='store_true',
                        help='Run ground state FD calculation. '
                        'Creates fd_gs.gpw.')
    parser.add_argument('--old-lcao-rt', action='store_true',
                        help='Run old implementation of LCAO time propagation'
                             '. Saves dipole moment file old_lcao_dm.out.')
    parser.add_argument('--new-lcao-rt', action='store_true',
                        help='Run new implementation of LCAO time propagation'
                             '. Saves dipole moment file new_lcao_dm.out.')
    parser.add_argument('--old-fd-rt', action='store_true',
                        help='Run old implementation of FD time propagation'
                             '. Saves dipole moment file old_fd_dm.out.')
    parser.add_argument('--new-fd-rt', action='store_true',
                        help='Run new implementation of FD time propagation'
                             '. Saves dipole moment file new_fd_dm.out.')
    parser.add_argument('--plot', action='store_true',
                        help='Plot dipole moments.')

    parsed = parser.parse_args()

    atoms = Atoms('H2', positions=[(0, 0, 0), (1, 0, 0)])
    atoms.center(vacuum=5)
    atoms.pbc = False

    kick_v = [1e-5, 0, 0]

    if parsed.lcao_gs or parsed.all:
        old_calc = GPAW(mode='lcao', basis='sz(dzp)', xc='LDA',
                        symmetry={'point_group': False},
                        txt='lcao.out', convergence={'density': 1e-12})
        atoms.calc = old_calc
        atoms.get_potential_energy()
        old_calc.write('lcao_gs.gpw', mode='all')

    if parsed.fd_gs or parsed.all:
        old_calc = GPAW(mode='fd', basis='sz(dzp)', xc='LDA',
                        symmetry={'point_group': False},
                        txt='fd.out', convergence={'density': 1e-12})
        atoms.calc = old_calc
        atoms.get_potential_energy()
        old_calc.write('fd_gs.gpw', mode='all')

    if parsed.old_lcao_rt or parsed.all:
        old_tddft = LCAOTDDFT('lcao_gs.gpw', propagator='ecn',
                              txt='/dev/null')
        DipoleMomentWriter(old_tddft, 'old_lcao_dm.out')
        old_tddft.absorption_kick(kick_v)
        old_tddft.propagate(10, 10)

    if parsed.old_fd_rt or parsed.all:
        old_tddft = TDDFT('fd_gs.gpw', propagator='ECN',
                          txt='/dev/null')
        DipoleMomentWriter(old_tddft, 'old_fd_dm.out')
        old_tddft.absorption_kick(kick_v)
        old_tddft.propagate(10, 10)

    def write_initial(fp, calc):
        dipolemoment_xv = [
            calc.calculate_dipole_moment(wfs)  # type: ignore
            for wfs in calc.state.ibzwfs]
        dm = np.sum(dipolemoment_xv, axis=0)
        norm = np.sum(calc.state.density.nct_aX.integrals)
        fp.write('%20.8lf %20.8le %22.12le %22.12le %22.12le\n' %
                 (0, norm, dm[0], dm[1], dm[2]))

    def write_result(fp, result):
        print(result)
        dm = result.dipolemoment
        fp.write('%20.8lf %20.8le %22.12le %22.12le %22.12le\n' %
                 (result.time, result.norm, dm[0], dm[1], dm[2]))

    if parsed.new_lcao_rt or parsed.all:
        new_tddft = RTTDDFT.from_dft_file('lcao_gs.gpw')

        dt = 10 * as_to_au * autime_to_asetime
        with open('new_lcao_dm.out', 'w') as fp:
            fp.write('# %15s %15s %22s %22s %22s\n' %
                     ('time', 'norm', 'dmx', 'dmy', 'dmz'))
            fp.write('# Start; Time = %.8lf\n' % 0)
            write_initial(fp, new_tddft)

            result = new_tddft.absorption_kick(kick_v)
            fp.write(f'# Kick {kick_v}; Time 0.0\n')
            write_result(fp, result)
            for result in new_tddft.ipropagate(dt, 10):
                write_result(fp, result)

    if parsed.new_fd_rt or parsed.all:
        new_tddft = RTTDDFT.from_dft_file('fd_gs.gpw')

        dt = 10 * as_to_au * autime_to_asetime
        with open('new_fd_dm.out', 'w') as fp:
            # TODO for some reason these are NDArrayReader objects
            for wfs in new_tddft.state.ibzwfs:
                wfs.psit_nX.data = wfs.psit_nX.data[:]
            fp.write('# %15s %15s %22s %22s %22s\n' %
                     ('time', 'norm', 'dmx', 'dmy', 'dmz'))
            fp.write('# Start; Time = %.8lf\n' % 0)
            write_initial(fp, new_tddft)

            result = new_tddft.absorption_kick(kick_v)
            fp.write(f'# Kick {kick_v}; Time 0.0\n')
            write_result(fp, result)
            for result in new_tddft.ipropagate(dt, 10):
                write_result(fp, result)
                fp.flush()

    if parsed.plot:
        import matplotlib.pyplot as plt
        for dmfile, label in [('old_lcao_dm.out', 'Old LCAO'),
                              ('new_lcao_dm.out', 'New LCAO'),
                              ('old_fd_dm.out', 'Old FD'),
                              ('new_fd_dm.out', 'New FD'),
                              ]:
            try:
                t, _, dmx, dmy, dmz = np.loadtxt(dmfile, unpack=True)
                plt.plot(t, dmx, label=label)
            except FileNotFoundError:
                pass
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
