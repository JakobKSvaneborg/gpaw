import numpy as np
from functools import cached_property
from ase import Atoms
from ase.units import Bohr
from types import SimpleNamespace

from gpaw.band_descriptor import BandDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.new import prod, zips
from gpaw.new.calculation import DFTCalculation
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.projections import Projections
from gpaw.pw.descriptor import PWDescriptor
from gpaw.utilities import pack_density
from gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions
from gpaw.utilities.timing import nulltimer


class PT:
    def __init__(self, ibzwfs, gd):
        self.ibzwfs = ibzwfs
        self.scale = 1 / gd.N_c.prod()

    def integrate(self, psit_nG, P_ani, q):
        self.ibzwfs.wfs_qs[q][0].pt_aiX._lfc.integrate(psit_nG, P_ani, q=0,
                                                       _scale=self.scale)


class FakeWFS:
    def __init__(self,
                 state,
                 setups,
                 comm,
                 occ,
                 hamiltonian,
                 atoms: Atoms):
        from gpaw.utilities.partition import AtomPartition
        self.timer = nulltimer
        self.setups = setups
        self.state = state
        self.hamiltonian = hamiltonian
        ibzwfs = self.state.ibzwfs
        self.kd = KPointDescriptor(ibzwfs.ibz.bz.kpt_Kc,
                                   ibzwfs.nspins)
        self.kd.set_symmetry(atoms,
                             ibzwfs.ibz.symmetries.symmetry)
        self.kd.set_communicator(ibzwfs.kpt_comm)
        self.bd = BandDescriptor(ibzwfs.nbands, ibzwfs.band_comm)
        self.grid = self.state.density.nt_sR.desc
        self.gd = self.grid._gd
        atomdist = self.state.density.D_asii.layout.atomdist
        self.atom_partition = AtomPartition(atomdist.comm, atomdist.rank_a)
        self.setups.set_symmetry(ibzwfs.ibz.symmetries.symmetry)
        self.occupations = occ
        self.nvalence = int(round(ibzwfs.nelectrons))
        assert self.nvalence == ibzwfs.nelectrons
        self.world = comm
        if ibzwfs.fermi_levels is not None:
            self.fermi_levels = ibzwfs.fermi_levels
            if len(self.fermi_levels) == 1:
                self.fermi_level = self.fermi_levels[0]
        self.nspins = ibzwfs.nspins
        self.dtype = ibzwfs.dtype
        wfs = ibzwfs.wfs_qs[0][0]
        self.pd = None
        self.basis_functions = getattr(wfs,  # dft.scf_loop.hamiltonian,
                                       'basis', None)
        if isinstance(wfs, PWFDWaveFunctions):
            if hasattr(wfs.psit_nX.desc, 'ecut'):
                self.mode = 'pw'
                self.ecut = wfs.psit_nX.desc.ecut
                self.pd = PWDescriptor(self.ecut,
                                       self.gd, self.dtype, self.kd)
                self.pwgrid = self.grid.new(dtype=self.dtype)
            else:
                self.mode = 'fd'
        else:
            self.mode = 'lcao'
            self.manytci = wfs.tci_derivatives.manytci
            if self.basis_functions is not None:
                self.ksl = SimpleNamespace(Mstart=self.basis_functions.Mstart,
                                           Mstop=self.basis_functions.Mstop)
        self.collinear = wfs.ncomponents < 4
        self.positions_set = True
        self.read_from_file_init_wfs_dm = False
        self.pt = PT(ibzwfs, self.gd)

    def apply_pseudo_hamiltonian(self, kpt, ham, a1, a2):
        a_nX = self.state.ibzwfs.wfs_qs[kpt.q][0].psit_nX
        self.hamiltonian.apply_local_potential(
            self.state.potential.vt_sR[kpt.s],
            a_nX.new(data=a1),
            a_nX.new(data=a2))

    @cached_property
    def work_array(self):
        return np.empty(
            (self.bd.mynbands,) + self.state.ibzwfs.get_max_shape(),
            complex)

    @cached_property
    def work_matrix_nn(self):
        from gpaw.matrix import Matrix
        return Matrix(
            self.bd.nbands, self.bd.nbands,
            self.dtype,
            dist=(self.bd.comm, self.bd.comm.size))

    def make_preconditioner(self, blocksize):
        return self.hamiltonian.create_preconditioner(blocksize)

    def _get_wave_function_array(self, u, n, realspace=True, periodic=False):
        assert realspace and not periodic
        psit_X = self.kpt_u[u].wfs.psit_nX[n]
        if hasattr(psit_X, 'ifft'):
            psit_R = psit_X.ifft(grid=self.pwgrid, periodic=True)
            psit_R.multiply_by_eikr(psit_X.desc.kpt_c)
            return psit_R.data
        return psit_X.data

    def get_wave_function_array(self, n, k, s, realspace=True, periodic=False):
        if self.mode == 'lcao':
            assert not realspace
            return self.kpt_qs[k][s].C_nM[n]
        assert realspace
        psit_X = self.kpt_qs[k][s].wfs.psit_nX[n]
        if self.mode == 'pw':
            psit_R = psit_X.ifft(grid=self.pwgrid, periodic=True)
            if not periodic:
                psit_R.multiply_by_eikr(psit_X.desc.kpt_c)
        else:
            psit_R = psit_X
            if periodic:
                psit_R.multiply_by_eikr(-psit_R.desc.kpt_c)
        return psit_R.data

    def collect_projections(self, k, s):
        return self.kpt_qs[k][s].projections.collect()

    def collect_eigenvalues(self, k, s):
        return self.state.ibzwfs.wfs_qs[k][s].eig_n.copy()

    @cached_property
    def kpt_u(self):
        return [kpt
                for kpt_s in self.kpt_qs
                for kpt in kpt_s]

    @cached_property
    def kpt_qs(self):
        ngpts = prod(self.gd.N_c)
        return [[KPT(wfs, self.atom_partition, ngpts, self.pd)
                 for wfs in wfs_s]
                for wfs_s in self.state.ibzwfs.wfs_qs]

    def integrate(self, a_nX, b_nX, gi):
        assert gi
        A_nX = self.grid.from_data(a_nX)
        B_nX = self.grid.from_data(b_nX)
        return A_nX.matrix_elements(B_nX).data


class KPT:
    def __init__(self, wfs, atom_partition, ngpts, pd):
        self.ngpts = ngpts
        self.wfs = wfs
        self.pd = pd

        I1 = 0
        nproj_a = []
        for a, shape in enumerate(wfs.P_ani.layout.shape_a):
            I2 = I1 + prod(shape)
            nproj_a.append(I2 - I1)
            I1 = I2

        self.projections = Projections(
            wfs.nbands,
            nproj_a,
            atom_partition,
            wfs.P_ani.comm,
            wfs.ncomponents < 4,
            wfs.spin,
            data=wfs.P_ani.data)
        self.eps_n = wfs.myeig_n
        self.s = wfs.spin if wfs.ncomponents < 4 else None
        self.k = wfs.k
        self.q = wfs.q
        self.weight = wfs.spin_degeneracy * wfs.weight
        self.f_n = wfs.myocc_n * self.weight
        self.P_ani = wfs.P_ani
        if isinstance(wfs, PWFDWaveFunctions):
            self.psit_nX = wfs.psit_nX
        else:
            self.C_nM = wfs.C_nM.data
            self.S_MM = wfs.S_MM.data
            self.P_aMi = wfs.P_aMi

    @property
    def psit_nG(self):
        a_nG = self.psit_nX.data
        if a_nG.ndim == 4:
            return a_nG
        return a_nG * self.ngpts

    @cached_property
    def psit(self):
        band_comm = self.psit_nX.comm
        return PlaneWaveExpansionWaveFunctions(
            self.wfs.nbands, self.pd, self.wfs.dtype,
            self.psit_nX.data * self.ngpts,
            kpt=self.q,
            dist=(band_comm, band_comm.size),
            spin=self.s,
            collinear=self.wfs.ncomponents != 4)


class FakeDensity:
    def __init__(self, dft: DFTCalculation):
        self.setups = dft.setups
        self.state = dft.state
        self.D_asii = self.state.density.D_asii
        self.atom_partition = dft._atom_partition
        self.interpolate = dft.pot_calc._interpolate_density
        self.nt_sR = self.state.density.nt_sR
        self.nt_sG = self.nt_sR.data
        self.gd = self.nt_sR.desc._gd
        self.finegd = dft.pot_calc.fine_grid._gd
        self._densities = dft.densities()
        self.ncomponents = len(self.nt_sG)
        self.nspins = self.ncomponents % 3
        self.collinear = self.ncomponents < 4

    @cached_property
    def D_asp(self):
        D_asp = self.setups.empty_atomic_matrix(self.ncomponents,
                                                self.atom_partition)
        D_asp.update({a: np.array([pack_density(D_ii) for D_ii in D_sii.real])
                      for a, D_sii in self.D_asii.items()})
        return D_asp

    @cached_property
    def nt_sg(self):
        return self.interpolate(self.nt_sR)[0].data

    def interpolate_pseudo_density(self):
        pass

    def get_all_electron_density(self, *, atoms, gridrefinement):
        n_sr = self._densities.all_electron_densities(
            grid_refinement=gridrefinement).scaled(1 / Bohr, Bohr**3)
        return n_sr.data, n_sr.desc._gd


class FakeHamiltonian:
    def __init__(self, state, pot_calc):
        self.pot_calc = pot_calc
        self.finegd = pot_calc.fine_grid._gd
        self.grid = state.potential.vt_sR.desc
        # self.e_total_free = dft.results.get('free_energy')
        self.e_xc = state.potential.energies['xc']

    def restrict_and_collect(self, vxct_sg):
        fine_grid = self.pot_calc.fine_grid
        vxct_sr = fine_grid.empty(len(vxct_sg))
        vxct_sr.data[:] = vxct_sg
        vxct_sR = self.grid.empty(len(vxct_sg))
        for vxct_r, vxct_R in zips(vxct_sr, vxct_sR):
            self.pot_calc.restrict(vxct_r, vxct_R)
        return vxct_sR.data

    @property
    def xc(self):
        return self.pot_calc.xc.xc
