import numpy as np

from ase import Atoms

from gpaw import FD, LCAO
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.calculator import GPAW as old_GPAW
from gpaw.tddft import TDDFT
from gpaw.tddft import DipoleMomentWriter as FDDipoleMomentWriter
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import (
    DipoleMomentWriter as LCAODipoleMomentWriter)
from gpaw.new.ase_interface import GPAW as new_GPAW
from gpaw.new.rttddft import RTTDDFT
from gpaw.tddft.units import as_to_au, autime_to_asetime


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='Demo script for new implementation of RT-TDDFT')
    parser.add_argument('--all', action='store_true', help='Run everything.')
    parser.add_argument('--old-lcao-gs', action='store_true',
                        help='Run old implementation of ground state LCAO '
                             'calculation. Creates old_lcao_gs.gpw.')
    parser.add_argument('--new-lcao-gs', action='store_true',
                        help='Run new implementation of ground state LCAO '
                             'calculation. Creates new_lcao_gs.gpw.')
    parser.add_argument('--old-fd-gs', action='store_true',
                        help='Run old implementation of ground state FD '
                             'calculation. Creates old_lcao_gs.gpw.')
    parser.add_argument('--new-fd-gs', action='store_true',
                        help='Run new implementation of ground state FD '
                             'calculation. Creates new_lcao_gs.gpw.')
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

    def assert_equal(a, b):
        from gpaw.core.matrix import Matrix
        from gpaw.core.atom_arrays import AtomArrays

        def extract(o):
            if (isinstance(o, Matrix) or isinstance(o, AtomArrays) or
                isinstance(o, UniformGridFunctions)):
                return o.data
            else:
                return o

        a = extract(a)
        b = extract(b)

        assert np.allclose(a, b), f'{str(a)} != {str(b)}'

    if parsed.old_lcao_gs or parsed.all:
        old_calc = old_GPAW(mode=LCAO(), basis='sz(dzp)', xc='LDA',
                            symmetry={'point_group': False},
                            txt='old_lcao.out', convergence={'density': 1e-12})
        atoms.calc = old_calc
        atoms.get_potential_energy()
        old_calc.write('old_lcao_gs.gpw', mode='all')

    if parsed.old_fd_gs or parsed.all:
        old_calc = old_GPAW(mode=FD(), basis='sz(dzp)', xc='LDA',
                            txt='old_fd.out', convergence={'density': 1e-12})
        atoms.calc = old_calc
        atoms.get_potential_energy()
        old_calc.write('old_fd_gs.gpw', mode='all')

    if parsed.new_lcao_gs or parsed.all:
        new_calc = new_GPAW(mode='lcao', basis='sz(dzp)', xc='LDA',
                            txt='new_lcao.out', force_complex_dtype=True,
                            convergence={'density': 1e-12})
        atoms.calc = new_calc
        atoms.get_potential_energy()
        new_calc.write('new_lcao_gs.gpw', mode='all')

        new_restart_calc = new_GPAW('new_lcao_gs.gpw')

        # Make sure that loading from disk works
        assert_equal(
            new_calc.dft.state.ibzwfs.wfs_qs[0][0].P_ain,
            new_restart_calc.dft.state.ibzwfs.wfs_qs[0][0].P_ain)

        assert_equal(
            new_calc.dft.state.ibzwfs.wfs_qs[0][0].C_nM,
            new_restart_calc.dft.state.ibzwfs.wfs_qs[0][0].C_nM)

    if parsed.new_fd_gs or parsed.all:
        new_calc = new_GPAW(mode='fd', basis='sz(dzp)', xc='LDA',
                            txt='new_fd.out', force_complex_dtype=True,
                            convergence={'density': 1e-12})
        atoms.calc = new_calc
        atoms.get_potential_energy()
        new_calc.write('new_fd_gs.gpw', mode='all')

        new_restart_calc = new_GPAW('new_fd_gs.gpw')

        # Make sure that loading from disk works
        assert_equal(
            new_calc.calculation.state.ibzwfs.wfs_qs[0][0].P_ani,
            new_restart_calc.calculation.state.ibzwfs.wfs_qs[0][0].P_ani)

        assert_equal(
            new_calc.calculation.state.ibzwfs.wfs_qs[0][0].psit_nX,
            new_restart_calc.calculation.state.ibzwfs.wfs_qs[0][0].psit_nX)

    if parsed.old_lcao_rt or parsed.all:
        old_tddft = LCAOTDDFT('old_lcao_gs.gpw', propagator='ecn',
                              txt='/dev/null')
        LCAODipoleMomentWriter(old_tddft, 'old_lcao_dm.out')
        old_tddft.absorption_kick(kick_v)
        old_tddft.propagate(10, 10)

    if parsed.old_fd_rt or parsed.all:
        old_tddft = TDDFT('old_fd_gs.gpw', propagator='ECN',
                          txt='/dev/null')
        FDDipoleMomentWriter(old_tddft, 'old_fd_dm.out')
        old_tddft.absorption_kick(kick_v)
        old_tddft.propagate(10, 10)

    def write_result(fp, result):
        print(result)
        dm = result.dipolemoment
        fp.write('%20.8lf %20.8le %22.12le %22.12le %22.12le\n' %
                 (result.time, 0, dm[0], dm[1], dm[2]))

    if parsed.new_lcao_rt or parsed.all:
        #  new_tddft = RTTDDFT.from_dft_calculation(new_calc)
        new_tddft = RTTDDFT.from_dft_file('new_lcao_gs.gpw')

        dt = 10 * as_to_au * autime_to_asetime
        with open('new_lcao_dm.out', 'w') as fp:
            result = new_tddft.absorption_kick(kick_v)
            fp.write(f'# Kick {kick_v}; Time 0.0')
            write_result(fp, result)
            for result in new_tddft.ipropagate(dt, 10):
                write_result(fp, result)

    if parsed.new_fd_rt or parsed.all:
        #  new_tddft = RTTDDFT.from_dft_calculation(new_calc)
        new_tddft = RTTDDFT.from_dft_file('new_fd_gs.gpw')

        dt = 10 * as_to_au * autime_to_asetime
        with open('new_fd_dm.out', 'w') as fp:
            # TODO for some reason these are NDArrayReader objects
            for wfs in new_tddft.state.ibzwfs:
                wfs.psit_nX.data = wfs.psit_nX.data[:]
            result = new_tddft.absorption_kick(kick_v)
            fp.write(f'# Kick {kick_v}; Time 0.0')
            write_result(fp, result)
            for result in new_tddft.ipropagate(dt, 10):
                write_result(fp, result)

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
