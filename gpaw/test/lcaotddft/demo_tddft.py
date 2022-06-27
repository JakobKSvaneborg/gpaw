import numpy as np

from ase import Atoms

from gpaw import LCAO
from gpaw.calculator import GPAW as old_GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
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
    parser.add_argument('--old-lcao-rt', action='store_true',
                        help='Run old implementation of LCAO time propagation'
                             '. Saves dipole moment file old_lcao_dm.out.')
    parser.add_argument('--new-lcao-rt', action='store_true',
                        help='Run new implementation of LCAO time propagation'
                             '. Saves dipole moment file new_lcao_dm.out.')
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
            if isinstance(o, Matrix):
                return o.data
            elif isinstance(o, AtomArrays):
                return o.data
            else:
                return o

        a = extract(a)
        b = extract(b)

        assert np.allclose(a, b), f'{str(a)} != {str(b)}'

    if parsed.old_lcao_gs or parsed.all:
        old_calc = old_GPAW(mode=LCAO(), basis='sz(dzp)', xc='LDA',
                            symmetry={'point_group': False},
                            txt='old.out', convergence={'density': 1e-12})
        atoms.calc = old_calc
        atoms.get_potential_energy()
        old_calc.write('old_lcao_gs.gpw', mode='all')

    if parsed.new_lcao_gs or parsed.all:
        new_calc = new_GPAW(mode='lcao', basis='sz(dzp)', xc='LDA',
                            txt='new.out', force_complex_dtype=True,
                            convergence={'density': 1e-12})
        atoms.calc = new_calc
        atoms.get_potential_energy()
        new_calc.write('new_lcao_gs.gpw', mode='all')

        new_restart_calc = new_GPAW('new_lcao_gs.gpw')

        assert_equal(
            new_calc.dft.state.ibzwfs.wfs_qs[0][0].P_ain,
            new_restart_calc.dft.state.ibzwfs.wfs_qs[0][0].P_ain)

        assert_equal(
            new_calc.dft.state.ibzwfs.wfs_qs[0][0].C_nM,
            new_restart_calc.dft.state.ibzwfs.wfs_qs[0][0].C_nM)

    if parsed.old_lcao_rt or parsed.all:
        old_tddft = LCAOTDDFT('old_lcao_gs.gpw', propagator='ecn',
                              txt='/dev/null')
        DipoleMomentWriter(old_tddft, 'old_lcao_dm.out')
        old_tddft.absorption_kick(kick_v)
        old_tddft.propagate(10, 10)
        # old_C_nM = old_tddft.wfs.kpt_u[0].C_nM
        # old_f_n = old_tddft.get_occupation_numbers()
        # old_rho_MM = old_C_nM.T.conj() @ (old_f_n[:, None] * old_C_nM)
        # print('rho_MM', old_rho_MM)

    if parsed.new_lcao_rt or parsed.all:
        #  new_tddft = RTTDDFT.from_dft_calculation(new_calc)
        new_tddft = RTTDDFT.from_dft_file('new_lcao_gs.gpw')

        new_tddft.absorption_kick(kick_v)
        dt = 10 * as_to_au * autime_to_asetime
        with open('new_lcao_dm.out', 'w') as fp:
            for result in new_tddft.ipropagate(dt, 10):
                dm = result.dipolemoment
                fp.write('%20.8lf %20.8le %22.12le %22.12le %22.12le\n' %
                         (result.time, 0, dm[0], dm[1], dm[2]))
                print(result)
        # wfs = new_tddft.state.ibzwfs.wfs_qs[0][0]
        # new_rho_MM = wfs.calculate_density_matrix()
        # print('rho_MM', new_rho_MM)

    if parsed.plot:
        import matplotlib.pyplot as plt
        for dmfile, label in [('old_lcao_dm.out', 'Old LCAO'),
                              ('new_lcao_dm.out', 'New LCAO'),
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
