from __future__ import annotations

from functools import cached_property
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.units import Bohr

from gpaw.old.band_descriptor import BandDescriptor
from gpaw.densities import Densities
from gpaw.fftw import MEASURE
from gpaw.new import prod, zips
from gpaw.new.density import Density
from gpaw.new.gpw import GPWFlags
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.potential import Potential
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.old.projections import Projections
from gpaw.old.pw.descriptor import PWDescriptor
from gpaw.utilities import pack_density
from gpaw.utilities.timing import nulltimer
from gpaw.old.wavefunctions.arrays import (PlaneWaveExpansionWaveFunctions,
                                           UniformGridWaveFunctions)


class PT:
    def __init__(self, ibzwfs):
        self.ibzwfs = ibzwfs

    def integrate(self, psit_nG, P_ani, q):
        pt_aiX = self.ibzwfs._wfs_u[self.ibzwfs.u_q[q]].pt_aiX
        pt_aiX._lazy_init()
        pt_aiX._lfc.integrate(psit_nG, P_ani, q=0)

    def add(self, psit_nG, c_axi, q):
        self.ibzwfs._wfs_u[self.ibzwfs.u_q[q]].pt_aiX._lfc.add(
            psit_nG, c_axi, q=0)

    def dict(self, shape):
        return self.ibzwfs._wfs_u[0].pt_aiX.empty(shape,
                                                  self.ibzwfs.band_comm)


class FakeWFS:
    def __init__(self,
                 ibzwfs,
                 density,
                 potential,
                 setups,
                 comm,
                 occ_calc,
                 hamiltonian,
                 atoms: Atoms,
                 scale_pw_coefs=False):
        from gpaw.utilities.partition import AtomPartition
        self.timer = nulltimer
        self.setups = setups
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        self.hamiltonian = hamiltonian
        ibz = ibzwfs.ibz
        self.kd = ibz._old_kd(ibzwfs.nspins, ibzwfs.kpt_comm)
        self.bd = BandDescriptor(ibzwfs.nbands, ibzwfs.band_comm)
        self.grid = density.nt_sR.desc
        self.gd = self.grid._gd
        atomdist = density.D_asii.layout.atomdist
        self.atom_partition = AtomPartition(atomdist.comm, atomdist.rank_a)
        # self.setups.set_symmetry(ibzwfs.ibz.symmetries.symmetry)
        self.occ_calc = occ_calc
        self.occupations = occ_calc.occ
        self.nvalence = int(round(density.nvalence))
        self.nvalence = density.nvalence
        # assert self.nvalence == density.nvalence
        self.world = comm
        if ibzwfs.fermi_levels is not None:
            self.fermi_levels = ibzwfs.fermi_levels
            if len(self.fermi_levels) == 1:
                self.fermi_level = self.fermi_levels[0]
        self.nspins = ibzwfs.nspins
        self.dtype = ibzwfs.dtype
        wfs = ibzwfs._wfs_u[0]
        self.pd = None
        self.basis_functions = getattr(wfs,  # dft.scf_loop.hamiltonian,
                                       'basis', None)
        if isinstance(wfs, PWFDWaveFunctions):
            if hasattr(wfs.psit_nX.desc, 'ecut'):
                self.mode = 'pw'
                self.ecut = wfs.psit_nX.desc.ecut
                self.pd = PWDescriptor(self.ecut,
                                       self.gd, self.dtype, self.kd, _new=True)
                self.pwgrid = self.grid.new(dtype=self.dtype)
            else:
                self.mode = 'fd'
        else:
            self.mode = 'lcao'
            self.manytci = wfs.tci_derivatives.manytci
            if self.basis_functions is not None:
                Mstart = self.basis_functions.Mstart
                Mstop = self.basis_functions.Mstop
                self.ksl = SimpleNamespace(Mstart=Mstart,
                                           Mstop=Mstop,
                                           using_blacs=False,
                                           world=self.world,
                                           nao=self.basis_functions.Mmax,
                                           mynao=Mstop - Mstart,
                                           block_comm=comm)
        self.collinear = wfs.ncomponents < 4
        self.positions_set = True
        self.read_from_file_init_wfs_dm = ibzwfs.read_from_file_init_wfs_dm

        self.pt = PT(ibzwfs)
        self.scalapack_parameters = (None, 1, 1, 128)
        self.ngpts = prod(self.gd.N_c)
        if self.mode == 'pw' and scale_pw_coefs:
            self.scale = self.ngpts
        else:
            self.scale = 1
        self.fftwflags = MEASURE

    def apply_pseudo_hamiltonian(self, kpt, ham, a1, a2):
        desc = self.ibzwfs._wfs_u[self.ibzwfs.u_q[kpt.q]].psit_nX.desc
        self.hamiltonian.apply(
            self.potential.vt_sR,
            None,
            self.ibzwfs,  # needed for hybrids
            getattr(ham, 'D_asii', None),  # needed for hybrids
            desc.from_data(data=a1),
            desc.from_data(data=a2),
            kpt.s)

    def calculate_occupation_numbers(self, fixed):
        self.ibzwfs.calculate_occs(
            self.occ_calc,
            fix_fermi_level=fixed)

    def empty(self, n, q):
        return np.empty(
            (n,) +
            self.ibzwfs._wfs_u[self.ibzwfs.u_q[q]].psit_nX.data.shape[1:],
            complex if self.mode == 'pw' else self.dtype)

    @cached_property
    def work_array(self):
        return np.empty(
            (self.bd.mynbands,) + self.ibzwfs.get_max_shape(),
            complex if self.mode == 'pw' else self.dtype)

    @cached_property
    def work_matrix_nn(self):
        from gpaw.old.matrix import Matrix
        return Matrix(
            self.bd.nbands, self.bd.nbands,
            dtype=self.dtype,
            dist=(self.bd.comm, self.bd.comm.size))

    @property
    def orthonormalized(self):
        return self.ibzwfs._wfs_u[0].orthonormalized

    def orthonormalize(self, kpt=None):
        if kpt is None:
            kpts = list(self.ibzwfs)
        else:
            kpts = [self.ibzwfs._get_wfs(kpt.k, kpt.s)]
        for wfs in kpts:
            wfs._P_ani = None
            wfs.orthonormalized = False
            wfs.orthonormalize()

    def make_preconditioner(self, blocksize):
        if self.mode == 'pw':
            from gpaw.old.wavefunctions.pw import Preconditioner
            return Preconditioner(self.pd.G2_qG, self.pd,
                                  _scale=self.ngpts**2)
        from gpaw.preconditioner import Preconditioner
        return Preconditioner(self.gd, self.hamiltonian.kin, self.dtype,
                              blocksize)

    def _get_wave_function_array(self, u, n, realspace=True, periodic=False):
        assert realspace and not periodic
        psit_X = self.kpt_u[u].wfs.psit_nX[n]
        if hasattr(psit_X, 'ifft'):
            psit_R = psit_X.ifft(grid=self.pwgrid, periodic=True)
            psit_R.multiply_by_eikr(psit_X.desc.kpt_c)
            return psit_R.data
        return psit_X.data

    def get_wave_function_array(self, n, k, s,
                                realspace=True,
                                periodic=False,
                                cut=False):
        assert not cut
        assert self.ibzwfs.band_comm.size == 1
        assert self.ibzwfs.kpt_comm.size == 1
        u = k * self.ibzwfs.nspins + s
        if self.mode == 'lcao':
            assert not realspace
            return self.kpt_u[u].C_nM[n]
        psit_X = self.kpt_u[u].wfs.psit_nX[n]
        if not realspace:
            return psit_X.data
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
        assert self.ibzwfs.kpt_comm.size == 1
        u = k * self.ibzwfs.nspins + s
        return self.kpt_u[u].projections.collect()

    def collect_eigenvalues(self, k, s):
        assert self.ibzwfs.kpt_comm.size == 1
        u = k * self.ibzwfs.nspins + s
        return self.ibzwfs._wfs_u[u].eig_n.copy()

    @cached_property
    def kpt_u(self):
        return [KPT(self.mode, wfs, self.atom_partition, self.scale,
                    self.pd, self.gd)
                for wfs in self.ibzwfs._wfs_u]

    @cached_property
    def _____kpt_qs(self):
        1 / 0
        return [[KPT(self.mode, wfs, self.atom_partition, self.scale,
                     self.pd, self.gd)
                 for wfs in wfs_s]
                for wfs_s in self.ibzwfs.wfs_qs]

    def integrate(self, a_nX, b_nX, global_integral):
        if self.mode == 'fd':
            return self.gd.integrate(a_nX, b_nX, global_integral)
        x = self.pd.integrate(a_nX, b_nX, global_integral)
        return self.ngpts**2 * x

    def calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        assert self.ibzwfs.band_comm.size == 1
        assert self.ibzwfs.kpt_comm.size == 1
        rho_MM = self.ibzwfs._wfs_u[0].calculate_density_matrix()
        return rho_MM

    def write_wave_functions(self, writer):
        flags = GPWFlags(precision='double',
                         include_wfs=True, include_projections=True)
        if self.ibzwfs.collinear:
            spin_k_shape = (self.ibzwfs.ncomponents, len(self.ibzwfs.ibz))
        else:
            spin_k_shape = (len(self.ibzwfs.ibz),)
        self.ibzwfs._write_wave_functions(writer, spin_k_shape, flags)

    def write_occupations(self, writer):
        _, occ_skn = self.ibzwfs.get_all_eigs_and_occs()
        if not self.ibzwfs.collinear:
            occ_skn = occ_skn[0]
        writer.write(occupations=occ_skn)


class KPT:
    def __init__(self, mode, wfs, atom_partition, scale, pd, gd):
        self.mode = mode
        self.scale = scale
        self.wfs = wfs
        self.pd = pd
        self.gd = gd

        try:
            I1 = 0
            nproj_a = []
            for a, shape in enumerate(wfs.P_ani.layout.shape_a):
                I2 = I1 + prod(shape)
                nproj_a.append(I2 - I1)
                I1 = I2
        except RuntimeError:
            pass
        else:
            self.projections = Projections(
                wfs.nbands,
                nproj_a,
                atom_partition,
                wfs.P_ani.comm,
                wfs.ncomponents < 4,
                wfs.spin,
                data=wfs.P_ani.data)

        self.s = wfs.spin if wfs.ncomponents < 4 else None
        self.k = wfs.k
        self.q = wfs.q
        self.weight = wfs.spin_degeneracy * wfs.weight
        self.weightk = wfs.weight
        if isinstance(wfs, PWFDWaveFunctions):
            self.psit_nX = wfs.psit_nX
        else:
            self.C_nM = wfs.C_nM.data
            self.S_MM = wfs.S_MM.data
            self.P_aMi = wfs.P_aMi
        if mode == 'fd':
            self.phase_cd = wfs.psit_nX.desc.phase_factor_cd
        else:
            self.phase_cd = None

    @property
    def P_ani(self):
        return self.wfs.P_ani

    @property
    def eps_n(self):
        return self.wfs.myeig_n

    @property
    def f_n(self):
        f_n = self.wfs.myocc_n * self.weight
        f_n.flags.writeable = False
        return f_n

    @f_n.setter
    def f_n(self, val):
        self.wfs.myocc_n[:] = val / self.weight

    @property
    def psit_nG(self):
        if not hasattr(self, 'psit_nX'):
            return None
        data = self.psit_nX.data
        if self.scale == 1:
            return data
        if 1:  # isinstance(data, np.ndarray):
            return data * self.scale
        data.scale *= self.scale
        return data

    @cached_property
    def psit(self):
        band_comm = self.psit_nX.comm
        if self.mode == 'pw':
            return PlaneWaveExpansionWaveFunctions(
                self.wfs.nbands, self.pd, self.wfs.dtype,
                self.psit_nG,
                kpt=self.q,
                dist=(band_comm, band_comm.size),
                spin=self.s,
                collinear=self.wfs.ncomponents != 4)
        return UniformGridWaveFunctions(
            self.wfs.nbands, self.gd, self.wfs.dtype,
            self.psit_nX.data,
            kpt=self.q,
            dist=(band_comm, band_comm.size),
            spin=self.s,
            collinear=self.wfs.ncomponents != 4)


class FakeDensity:
    def __init__(self,
                 ibzwfs: IBZWaveFunctions,
                 density: Density,
                 potential: Potential,
                 pot_calc: PotentialCalculator,
                 densities: Densities | None = None):
        self.setups = pot_calc.setups
        self.D_asii = density.D_asii
        self.atom_partition = self._atom_partition
        self.ccc_aL = density.calculate_compensation_charge_coefficients()
        try:
            # mypy complains about these missing from PotentialCalculator
            self.interpolate = pot_calc._interpolate_density  # type: ignore
            self.finegd = pot_calc.fine_grid._gd  # type: ignore
            self.fine_grid = pot_calc.fine_grid  # type: ignore
        except AttributeError:
            pass
        try:
            # Only in LCAO
            self.ghat_aLr = pot_calc.ghat_aLr  # type: ignore
        except AttributeError:
            pass
        self.nt_sR = density.nt_sR
        self.nt_sG = self.nt_sR.data
        self.gd = self.nt_sR.desc._gd
        self._densities = densities
        self.ncomponents = len(self.nt_sG)
        self.nspins = self.ncomponents % 3
        self.collinear = self.ncomponents < 4

    @cached_property
    def _atom_partition(self):
        from gpaw.utilities.partition import AtomPartition
        atomdist = self.D_asii.layout.atomdist
        return AtomPartition(atomdist.comm, atomdist.rank_a)

    @cached_property
    def D_asp(self):
        D_asp = self.setups.empty_atomic_matrix(self.ncomponents,
                                                self.atom_partition)
        D_asp.update({a: np.array([pack_density(D_ii) for D_ii in D_sii.real])
                      for a, D_sii in self.D_asii.items()})
        return D_asp

    @cached_property
    def nt_sg(self):
        # Intepolate density
        nt_sr = self.interpolate(self.nt_sR)[0]

        # Compute pseudo charge
        pseudo_charge = nt_sr.integrate().sum()
        comp_charge = (4 * np.pi)**0.5 * sum(float(ccc_L[0])
                                             for ccc_L in self.ccc_aL.values())
        comp_charge = self.ccc_aL.layout.atomdist.comm.sum_scalar(comp_charge)

        # Normalize
        nt_sr.data *= -comp_charge / pseudo_charge
        return nt_sr.data

    @cached_property
    def rhot_g(self):
        rhot_g = self.fine_grid.empty()
        rhot_g.data[:] = self.nt_sg.sum(axis=0)
        self.ghat_aLr.add_to(rhot_g, self.ccc_aL)
        return rhot_g.data

    def interpolate_pseudo_density(self):
        pass

    def get_all_electron_density(self, *, atoms, gridrefinement):
        if self._densities is None:
            raise AttributeError('densities not set')
        n_sr = self._densities.all_electron_densities(
            grid_refinement=gridrefinement).scaled(1 / Bohr, Bohr**3)
        return n_sr.data, n_sr.desc._gd


class FakePoisson:

    def get_description(self):
        return ''


class FakeHamiltonian:
    def __init__(self,
                 ibzwfs: IBZWaveFunctions,
                 density: Density,
                 potential: Potential,
                 pot_calc: PotentialCalculator,
                 e_band=np.nan,
                 e_kinetic0=np.nan,
                 e_coulomb=np.nan,
                 e_total_free=np.nan,
                 e_zero=np.nan,
                 e_external=np.nan,
                 e_entropy=np.nan,
                 e_extrapolation=np.nan,
                 e_xc=np.nan):
        self.pot_calc = pot_calc
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        self.poisson = FakePoisson()
        try:
            self.finegd = self.pot_calc.fine_grid._gd  # type: ignore
        except AttributeError:
            pass
        self.grid = potential.vt_sR.desc
        self.e_total_free = e_total_free
        self.e_xc = e_xc
        self.e_kinetic0 = e_kinetic0
        self.e_coulomb = e_coulomb
        self.e_external = e_external
        self.e_zero = e_zero
        self.e_entropy = e_entropy
        self.e_extrapolation = e_extrapolation
        self.e_band = e_band

    def update(self, dens, wfs, kin_en_using_band=True):
        self.potential, _ = self.pot_calc.calculate(
            self.density, self.ibzwfs, self.potential.vHt_x)

        energies = self.potential.energies
        self.e_xc = energies['xc']
        self.e_coulomb = energies['coulomb']
        self.e_zero = energies['zero']
        self.e_external = energies['external']

        if kin_en_using_band:
            self.e_kinetic0 = energies['kinetic']
        else:
            self.e_kinetic0 = self.ibzwfs.calculate_kinetic_energy(
                wfs.hamiltonian, self.density)
            self.ibzwfs.energies['exx_kinetic'] = 0.0
            energies['kinetic'] = self.e_kinetic0

    def get_energy(self, e_entropy, wfs, kin_en_using_band=True, e_sic=None):
        self.e_band = self.ibzwfs.energies['band']
        if kin_en_using_band:
            self.e_kinetic = self.e_kinetic0 + self.e_band
        else:
            self.e_kinetic = self.e_kinetic0
        self.e_entropy = e_entropy
        if 0:
            print(self.e_kinetic0,
                  self.e_band,
                  self.e_coulomb,
                  self.e_external,
                  self.e_zero,
                  self.e_xc,
                  self.e_entropy)
        self.e_total_free = (self.e_kinetic + self.e_coulomb +
                             self.e_external + self.e_zero + self.e_xc +
                             self.e_entropy)

        if e_sic is not None:
            self.e_sic = e_sic
            self.e_total_free += e_sic

        self.e_total_extrapolated = (
            self.e_total_free +
            self.ibzwfs.energies['extrapolation'])

        return self.e_total_free

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

    @property
    def dH_asp(self):
        from gpaw.utilities.partition import AtomPartition
        dH_asp = self.potential.dH_asii.to_cpu().to_lower_triangle().gather()
        atomdist = self.potential.dH_asii.layout.atomdist
        atom_partition = AtomPartition(atomdist.comm, atomdist.rank_a)
        dH_asp.partition = atom_partition
        return dH_asp

    def dH(self, P, out):
        for a, I1, I2 in P.indices:
            dH_ii = self.potential.dH_asii[a][P.spin]
            out.array[:, I1:I2] = np.dot(P.array[:, I1:I2], dH_ii)
