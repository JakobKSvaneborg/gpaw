from __future__ import annotations

from collections.abc import Callable, Generator
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from ase.io.ulm import Writer
from ase.units import Bohr, Ha

from gpaw.gpu import as_np
from gpaw.gpu.mpi import CuPyMPI
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import zips
from gpaw.new.brillouin import IBZ
from gpaw.new.c import GPU_AWARE_MPI
from gpaw.new.potential import Potential
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.timer import trace
from gpaw.new.wave_functions import WaveFunctions
from gpaw.typing import Array1D, Array2D, Self
from gpaw.utilities import as_complex_dtype, pack_density

if TYPE_CHECKING:
    from gpaw.new.density import Density

WFT = TypeVar('WFT', bound=WaveFunctions)


class IBZWaveFunctions(Generic[WFT]):
    def __init__(self,
                 ibz: IBZ,
                 *,
                 ncomponents: int,
                 wfs_u: list[WFT],
                 kpt_comm: MPIComm = serial_comm,
                 kpt_band_comm: MPIComm = serial_comm,
                 comm: MPIComm = serial_comm):
        """Collection of wave function objects for k-points in the IBZ."""
        self.ibz = ibz
        self.kpt_comm = kpt_comm
        self.kpt_band_comm = kpt_band_comm
        self.comm = comm
        self.ncomponents = ncomponents
        self.collinear = (ncomponents != 4)
        self.spin_degeneracy = ncomponents % 2 + 1
        self.nspins = ncomponents % 3

        # kpt_comm ranks:
        self.rank_ks = ibz.ranks(kpt_comm, self.nspins)
        self.nu_r = np.bincount(self.rank_ks.ravel())

        self._wfs_u = wfs_u
        wfs = wfs_u[0]
        self.u0 = wfs.k * self.nspins + wfs.spin
        self.u_q = {wfs.q: u for u, wfs in enumerate(wfs_u)}

        self.band_comm = wfs.band_comm
        self.domain_comm = wfs.domain_comm
        self.dtype = wfs.dtype
        self.nbands = wfs.nbands

        self.fermi_levels: Array1D | None = None  # hartree

        self.xp = wfs.xp
        if self.xp is not np:
            if not GPU_AWARE_MPI:
                self.kpt_comm = CuPyMPI(self.kpt_comm)  # type: ignore

        self.move_wave_functions: Callable[..., None] = lambda *args: None

        self.read_from_file_init_wfs_dm = False

    @classmethod
    def create(cls,
               *,
               ibz: IBZ,
               ncomponents: int,
               create_wfs_func,
               kpt_comm: MPIComm = serial_comm,
               kpt_band_comm: MPIComm = serial_comm,
               comm: MPIComm = serial_comm,
               ) -> Self:
        nspins = ncomponents % 3
        rank_ks = ibz.ranks(kpt_comm, nspins)

        wfs_u: list[WFT] = []
        q = 0
        for k in range(len(ibz)):
            for spin in range(nspins):
                if kpt_comm.rank == rank_ks[k, spin]:
                    wfs = create_wfs_func(spin, q, k,
                                          ibz.kpt_kc[k], ibz.weight_k[k])
                    wfs_u.append(wfs)
                    if spin == nspins - 1:
                        q += 1
        return cls(ibz,
                   ncomponents=ncomponents,
                   wfs_u=wfs_u,
                   kpt_comm=kpt_comm,
                   kpt_band_comm=kpt_band_comm,
                   comm=comm)

    @cached_property
    def mode(self):
        wfs = self._wfs_u[0]
        if isinstance(wfs, PWFDWaveFunctions):
            if hasattr(wfs.psit_nX.desc, 'ecut'):
                return 'pw'
            return 'fd'
        return 'lcao'

    def has_wave_functions(self) -> bool:
        raise NotImplementedError

    def get_max_shape(self, global_shape: bool = False) -> tuple[int, ...]:
        """Find the largest wave function array shape.

        For a PW-calculation, this shape could depend on k-point.
        """
        if global_shape:
            shape = np.array(max(wfs.array_shape(global_shape=True)
                                 for wfs in self))
            self.kpt_comm.max(shape)
            return tuple(shape)
        return max(wfs.array_shape() for wfs in self)

    @property
    def fermi_level(self) -> float:
        fl = self.fermi_levels
        assert fl is not None and len(fl) == 1
        return fl[0]

    def summary(self, log):
        shape = self.get_max_shape(global_shape=True)
        wfs = self._wfs_u[0]
        nbytes = (len(self.ibz) *
                  self.nbands *
                  self.nspins *
                  wfs.bytes_per_band)
        ncores = (self.kpt_comm.size *
                  self.domain_comm.size *
                  self.band_comm.size)
        nproj = sum(len(setup.pt_j) for setup in wfs.setups)
        nbytesproj = bytes_for_projectors(self)
        if nbytesproj != -1:
            projectors_text = (
                f'  Projectors: {nbytesproj:_} bytes '
                f'({nbytesproj * self.band_comm.size // ncores:_} per core)\n')
        else:
            projectors_text = ''

        self.ibz.symmetries.summary(log)
        self.ibz.summary(log)
        log(f'{wfs._short_string(shape)}\n'
            f'Spin-components: {self.ncomponents}'
            ' (' + ('' if self.collinear else 'non-') + 'collinear spins)\n'
            f'Bands:           {self.nbands}\n'
            f'Projectors:      {nproj}\n'
            f'Spin-degeneracy: {self.spin_degeneracy}\n'
            f'Datatype:        {self.dtype}\n\n'
            'Memory:\n'
            f'  Storage: {"CPU" if self.xp is np else "GPU"}\n'
            f'  Wave functions: {nbytes:_} bytes '
            f'({nbytes // ncores:_} per core)\n' +
            projectors_text +
            '\nParallelization:\n'
            f'  kpt:    {self.kpt_comm.size}\n'
            f'  domain: {self.domain_comm.size}\n'
            f'  band:   {self.band_comm.size}\n')

    def __iter__(self) -> Generator[WFT]:
        yield from self._wfs_u

    def padded(self):
        for wfs in self._wfs_u:
            yield wfs
        npad = self.nu_r.max() - self.nu_r[self.kpt_comm.rank]
        w = wfs.weight
        wfs.weight = 0.0
        for _ in range(npad):
            yield wfs
        wfs.weight = w

    def move(self, relpos_ac, atomdist):
        self.ibz.symmetries.check_positions(relpos_ac)
        self.make_sure_wfs_are_read_from_gpw_file()
        for wfs in self:
            wfs.move(relpos_ac, atomdist, self.move_wave_functions)

    def orthonormalize(self, work_array_nX: np.ndarray = None):
        for wfs in self:
            wfs.orthonormalize(work_array_nX)

    @trace
    def calculate_occs(self,
                       occ_calc,
                       nelectrons: float,
                       fix_fermi_level=False) -> tuple[float, float, float]:
        degeneracy = self.spin_degeneracy

        # u index is q and s combined
        occ_un, fermi_levels, e_entropy = occ_calc.calculate(
            nelectrons=nelectrons / degeneracy,
            eigenvalues=[wfs.eig_n * Ha for wfs in self],
            weights=[wfs.weight for wfs in self],
            spins=[wfs.spin for wfs in self],
            fermi_levels_guess=(None
                                if self.fermi_levels is None else
                                self.fermi_levels * Ha),
            fix_fermi_level=fix_fermi_level)

        if not fix_fermi_level:
            self.fermi_levels = np.array(fermi_levels) / Ha
        else:
            assert self.fermi_levels is not None

        for occ_n, wfs in zips(occ_un, self):
            wfs._occ_n = occ_n

        e_entropy *= degeneracy / Ha
        e_band = 0.0
        for wfs in self:
            e_band += wfs.occ_n @ wfs.eig_n * wfs.weight * degeneracy
        e_band = self.kpt_comm.sum_scalar(float(e_band))  # XXX CPU float?

        return e_band, e_entropy, e_entropy * occ_calc.extrapolate_factor

    def add_to_density(self, nt_sR, D_asii) -> None:
        """Compute density and add to ``nt_sR`` and ``D_asii``."""
        for wfs in self:
            wfs.add_to_density(nt_sR, D_asii)

        # This should be done in a more efficient way!!!
        # Also: where do we want the density?
        self.kpt_comm.sum(nt_sR.data)
        self.kpt_comm.sum(D_asii.data)
        self.band_comm.sum(nt_sR.data)
        self.band_comm.sum(D_asii.data)

    def normalize_density(self, density: Density) -> None:
        pass  # overwritten in LCAOIBZWaveFunctions class

    def add_to_ked(self, taut_sR) -> None:
        for wfs in self:
            wfs.add_to_ked(taut_sR)
        self.kpt_comm.sum(taut_sR.data)
        self.band_comm.sum(taut_sR.data)

    def get_all_electron_wave_function(self,
                                       band,
                                       kpt=0,
                                       spin=0,
                                       grid_spacing=0.05,
                                       skip_paw_correction=False):
        wfs = self.get_wfs_on_master(kpt=kpt, spin=spin, n1=band, n2=band + 1)
        if wfs is None:
            return None
        assert isinstance(wfs, PWFDWaveFunctions)
        psit_X = wfs.psit_nX[0].to_pbc_grid()
        grid = psit_X.desc.uniform_grid_with_grid_spacing(grid_spacing)
        psi_r = psit_X.interpolate(grid=grid)

        if not skip_paw_correction:
            dphi_aj = wfs.setups.partial_wave_corrections()
            dphi_air = grid.atom_centered_functions(dphi_aj, wfs.relpos_ac)
            dphi_air.add_to(psi_r, wfs.P_ani[:, 0])

        return psi_r

    def _get_wfs(self, kpt, spin):
        u = kpt * self.nspins + spin - self.u0
        assert u >= 0, (kpt, spin, self.nspins, self.u0)
        return self._wfs_u[u]

    def get_wfs_on_master(self,
                          *,
                          kpt: int = 0,
                          spin: int = 0,
                          n1=0,
                          n2=0):
        krank = self.rank_ks[kpt][spin]
        if krank == self.kpt_comm.rank:
            wfs = self._get_wfs(kpt, spin)
            wfs2 = wfs.collect_bands_and_domain(n1, n2)
            if krank == 0:
                return wfs2
            if wfs2 is not None:
                wfs2.send(0, self.kpt_comm)
            return
        if self.comm.rank == 0:
            return self._wfs_u[0].receive(krank, self.kpt_comm)
        return None

    def gather(self):
        # extract sizes from old wfs
        ncomponents = self.ncomponents
        nspins = ncomponents % 3
        ibz = self.ibz

        # gather wfs on master
        wfs_u = []
        for k in range(len(ibz)):
            for spin in range(nspins):
                wfs = self.get_wfs_on_master(kpt=k, spin=spin)
                wfs_u.append(wfs)

        fermi_levels = self.fermi_levels

        return ibz, ncomponents, wfs_u, fermi_levels

    def get_eigs_and_occs(self, kpt=0, spin=0):
        if self.domain_comm.rank == 0 and self.band_comm.rank == 0:
            rank = self.rank_ks[kpt, spin]
            if rank == self.kpt_comm.rank:
                wfs = self._get_wfs(kpt, spin)
                if rank == 0:
                    return wfs.eig_n, wfs._occ_n
                self.kpt_comm.send(wfs.eig_n, 0)
                self.kpt_comm.send(wfs._occ_n, 0)
            elif self.kpt_comm.rank == 0:
                eig_n = np.empty(self.nbands)
                occ_n = np.empty(self.nbands)
                self.kpt_comm.receive(eig_n, rank)
                self.kpt_comm.receive(occ_n, rank)
                return eig_n, occ_n
        return np.zeros(0), np.zeros(0)

    def get_all_eigs_and_occs(self, broadcast=False):
        nkpts = len(self.ibz)
        mynbands = self.nbands if self.comm.rank == 0 or broadcast else 0
        eig_skn = np.empty((self.nspins, nkpts, mynbands))
        occ_skn = np.empty((self.nspins, nkpts, mynbands))
        for k in range(nkpts):
            for s in range(self.nspins):
                eig_n, occ_n = self.get_eigs_and_occs(k, s)
                if self.comm.rank == 0:
                    eig_skn[s, k, :] = eig_n
                    occ_skn[s, k, :] = occ_n
        if broadcast:
            self.comm.broadcast(eig_skn, 0)
            self.comm.broadcast(occ_skn, 0)
        return eig_skn, occ_skn

    def forces(self,
               potential: Potential,
               hamiltonian,
               D_asii) -> Array2D:
        self.make_sure_wfs_are_read_from_gpw_file()
        F_av = self.xp.zeros((len(potential.dH_asii), 3))
        for wfs in self:
            wfs.force_contribution(potential, F_av)
        hamiltonian.update_wave_functions(self, forces=True)
        for wfs in self.padded():
            if isinstance(wfs, PWFDWaveFunctions):
                hamiltonian.apply_orbital_dependent(
                    self, D_asii, wfs.psit_nX, wfs.spin,
                    calculate_energy=False, F_av=F_av)
        self.kpt_band_comm.sum(F_av)
        return F_av

    def write(self, writer: Writer, flags) -> None:
        """Write fermi-level(s), eigenvalues, occupation numbers, ...

        ... k-points, symmetry information, projections and possibly
        also the wave functions.
        """
        eig_skn, occ_skn = self.get_all_eigs_and_occs()
        if not self.collinear:
            eig_skn = eig_skn[0]
            occ_skn = occ_skn[0]
        assert self.fermi_levels is not None
        writer.write(fermi_levels=self.fermi_levels * Ha,
                     eigenvalues=eig_skn * Ha,
                     occupations=occ_skn)
        ibz = self.ibz
        writer.child('kpts').write(
            atommap=ibz.symmetries.atommap_sa,
            bz2ibz=ibz.bz2ibz_K,
            bzkpts=ibz.bz.kpt_Kc,
            ibzkpts=ibz.kpt_kc,
            rotations=ibz.symmetries.rotation_scc,
            translations=ibz.symmetries.translation_sc,
            weights=ibz.weight_k)

        nproj = self._wfs_u[0].P_ani.layout.size

        spin_k_shape: tuple[int, ...]
        proj_shape: tuple[int, ...]

        if self.collinear:
            spin_k_shape = (self.ncomponents, len(ibz))
            proj_shape = (self.nbands, nproj)
        else:
            spin_k_shape = (len(ibz),)
            proj_shape = (self.nbands, 2, nproj)

        if flags.include_projections:
            proj_dtype = flags.storage_dtype(self.dtype)
            writer.add_array('projections', spin_k_shape + proj_shape,
                             proj_dtype)
            for spin in range(self.nspins):
                for k, rank in enumerate(self.rank_ks[:, spin]):
                    if rank == self.kpt_comm.rank:
                        wfs = self._get_wfs(k, spin)
                        P_ani = wfs.P_ani.to_cpu().gather()  # gather atoms
                        if P_ani is not None:
                            P_nI = P_ani.matrix.gather()  # gather bands
                            if P_nI is not None:
                                if rank == 0:
                                    writer.fill(P_nI.data.reshape(
                                        proj_shape).astype(proj_dtype))
                                else:
                                    self.kpt_comm.send(P_nI.data, 0)
                    elif self.comm.rank == 0:
                        data = np.empty(proj_shape, self.dtype)
                        self.kpt_comm.receive(data, rank)
                        writer.fill(data.astype(proj_dtype))

        if flags.include_wfs and self.has_wave_functions():
            self._write_wave_functions(writer, spin_k_shape, flags)

    def _write_wave_functions(self, writer, spin_k_shape, flags):
        self.make_sure_wfs_are_read_from_gpw_file()
        # We collect all bands to master.  This may have to be changed
        # to only one band at a time XXX
        xshape = self.get_max_shape(global_shape=True)
        shape = spin_k_shape + (self.nbands,) + xshape
        dtype = (as_complex_dtype(self.dtype) if self.mode == 'pw'
                 else self.dtype)
        dtype_write = flags.storage_dtype(dtype)
        c = 1.0 if self.mode == 'lcao' else Bohr**-1.5

        writer.add_array('coefficients', shape, dtype=dtype_write)
        buf_nX = np.empty((self.nbands,) + xshape, dtype=dtype)

        for spin in range(self.nspins):
            for k, rank in enumerate(self.rank_ks[:, spin]):
                if rank == self.kpt_comm.rank:
                    wfs = self._get_wfs(k, spin)
                    coef_nX = wfs.gather_wave_function_coefficients()
                    if wfs.domain_band_comm.rank == 0:
                        coef_nX = as_np(coef_nX)
                        if self.mode == 'pw':
                            x = coef_nX.shape[-1]
                            if x < xshape[-1]:
                                # For PW-mode, we may need to zero-pad the
                                # plane-wave coefficient up to the maximum
                                # for all k-points:
                                buf_nX[..., :x] = coef_nX
                                buf_nX[..., x:] = 0.0
                                coef_nX = buf_nX
                        if rank == 0:
                            writer.fill(flags.to_storage_dtype(coef_nX * c))
                        else:
                            self.kpt_comm.send(coef_nX, 0)
                elif self.comm.rank == 0:
                    self.kpt_comm.receive(buf_nX, rank)
                    writer.fill(flags.to_storage_dtype(buf_nX * c))

    def write_summary(self, log):
        fl = self.fermi_levels * Ha
        if len(fl) == 1:
            log(f'\nFermi level: {fl[0]:.3f}')
        else:
            log(f'\nFermi levels: {fl[0]:.3f}, {fl[1]:.3f}')

        ibz = self.ibz

        eig_skn, occ_skn = self.get_all_eigs_and_occs()

        if self.comm.rank != 0:
            return

        eig_skn *= Ha

        D = self.spin_degeneracy
        nbands = eig_skn.shape[2]

        k = 0  # only show one k-point

        # First, last and +-8 bands window around Fermi level:
        rows = []
        skipping = False
        if self.nspins == 1:
            header = ['band index', 'eig [eV]', f'occ [0-{D}]']
            eig_n = eig_skn[0, k]
            n0 = (eig_n < fl[0]).sum() - 0.5
            for n, (e, f) in enumerate(zips(eig_n, occ_skn[0, k])):
                if n == 0 or abs(n - n0) < 8 or n == nbands - 1:
                    rows.append([f'{n:4}', f'{e:13.3f}', f'{D * f:9.3f}'])
                    skipping = False
                else:
                    if not skipping:
                        rows.append(['...', '', ''])
                        skipping = True
        else:
            header = ['band index',
                      'eig [eV]', 'occ [0-1]',
                      'eig [eV]', 'occ [0-1]']
            eig1_n, eig2_n = eig_skn[:, k]
            n0 = ((eig1_n < fl[0]).sum() / 2 +
                  (eig2_n < fl[-1]).sum() / 2 - 0.5)
            for n, (e1, f1, e2, f2) in enumerate(zips(eig1_n,
                                                      occ_skn[0, k],
                                                      eig2_n,
                                                      occ_skn[1, k])):
                if n == 0 or abs(n - n0) < 8 or n == nbands - 1:
                    rows.append([f'{n:4}',
                                 f'{e1:13.3f}', f'{f1:9.3f}',
                                 f'{e2:13.3f}', f'{f2:9.3f}'])
                    skipping = False
                else:
                    if not skipping:
                        rows.append(['...', '', '', '', ''])
                        skipping = True

        x, y, z = ibz.kpt_kc[k]
        title = f'Eigenvalues  # (at kpt = [{x:.3f}, {y:.3f}, {z:.3f}])'
        log.table(title, header, rows)

        try:
            from ase.dft.bandgap import GapInfo
        except ImportError:
            log('No gapinfo -- requires new ASE')
            return

        try:
            log()
            fermilevel = fl[0]
            gapinfo = GapInfo(eigenvalues=eig_skn - fermilevel)
            log(gapinfo.description(ibz_kpoints=ibz.kpt_kc))
        except ValueError:
            # Maybe we only have the occupied bands and no empty bands
            log('Could not find a gap')

    def make_sure_wfs_are_read_from_gpw_file(self):
        for wfs in self:
            psit_nX = getattr(wfs, 'psit_nX', None)
            if psit_nX is None:
                return
            if hasattr(psit_nX.data, 'fd'):  # fd=file-descriptor
                self.read_from_file_init_wfs_dm = True
                psit_nX.data = np.ascontiguousarray(psit_nX.data[:])  # read

    def get_homo_lumo(self, spin: int = None) -> Array1D:
        """Return HOMO and LUMO eigenvalues."""
        if self.ncomponents == 1:
            assert spin == 0 or spin is None
            spin = 0
        elif self.ncomponents == 2:
            if spin is None:
                h0, l0 = self.get_homo_lumo(0)
                h1, l1 = self.get_homo_lumo(1)
                return np.array([max(h0, h1), min(l0, l1)])
        else:
            assert spin == 0 or spin is None
            spin = 0

        nocc = 0.0
        for wfs in self:
            if wfs.spin == spin:
                nocc += wfs.occ_n.sum() * wfs.weight
        nocc = self.kpt_comm.sum_scalar(nocc)
        n = int(round(nocc))

        homo = -np.inf
        if n > 0:
            for wfs in self:
                if wfs.spin == spin:
                    homo = max(homo, wfs.eig_n[n - 1])
        homo = self.kpt_comm.max_scalar(homo)

        lumo = np.inf
        if n < self.nbands:
            for wfs in self:
                if wfs.spin == spin:
                    lumo = min(lumo, wfs.eig_n[n])
        lumo = self.kpt_comm.min_scalar(lumo)

        return np.array([homo, lumo])

    def calculate_kinetic_energy(self,
                                 hamiltonian,
                                 density: Density) -> float:
        e_kin = 0.0
        for wfs in self:
            e_kin += hamiltonian.calculate_kinetic_energy(wfs, skip_sum=True)
        e_kin = self.comm.sum_scalar(e_kin)

        # PAW corrections:
        e_kin_paw = 0.0
        for a, D_sii in density.D_asii.items():
            setup = wfs.setups[a]
            D_p = pack_density(D_sii.real[:density.ndensities].sum(0))
            e_kin_paw += setup.K_p @ D_p + setup.Kc
        e_kin_paw = density.grid.comm.sum_scalar(e_kin_paw)

        return e_kin + e_kin_paw

    def number_of_occupied_bands(self,
                                 tolerance: float = 1e-5) -> int:
        nocc = 0
        for wfs in self:
            nocc = max(nocc, int((wfs.occ_n > tolerance).sum()))
        return int(self.kpt_comm.max_scalar(nocc))


def bytes_for_projectors(ibzwfs: IBZWaveFunctions) -> int:
    """Calculate number of bytest used for PAW projectors.

    Without creating them!
    """
    if ibzwfs.mode != 'pw':
        # TODO ...
        return -1
    wfs = ibzwfs._wfs_u[0]
    setups = wfs.setups
    nunique = 0
    lmax = 0
    for setup in setups.setups.values():
        nunique += len(setup.pt_j)
        lmax = max(max((pt.l for pt in setup.pt_j), default=0), lmax)
    return len(ibzwfs.ibz) * wfs.bytes_per_band * (
        (lmax + 1)**2 +  # Y_LG
        nunique +  # f_sG
        len(setups))  # emiGR_Ga
