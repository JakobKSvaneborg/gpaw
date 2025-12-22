from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from time import ctime, time

import numpy as np
from ase.dft import monkhorst_pack
from ase.units import Bohr, Hartree
from scipy.linalg import eigh

from gpaw.blacs import BlacsDescriptor, BlacsGrid, Redistributor
from gpaw.mpi import serial_comm, world
from gpaw.old.kpt_descriptor import KPointDescriptor
from gpaw.response import ResponseContext
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
from gpaw.response.context import timer
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.df import Chi0DysonEquations, write_response_function
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.groundstate import CellDescriptor
from gpaw.response.pair import KPointPairFactory, get_gs_and_context
from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.qpd import SingleQPWDescriptor
from gpaw.response.screened_interaction import (GammaIntegrationMode,
                                                initialize_w_calculator)
from gpaw.utilities.elpa import LibElpa


def decide_whether_tammdancoff(val_m, con_m):
    for n in val_m:
        if n in con_m:
            return False
    return True


@dataclass
class BSEMatrix:
    df_S: np.ndarray
    H_sS: np.ndarray
    deps_S: np.ndarray
    deps_max: float
    # Mapping from reduced to full indices (for early filtering)
    active_S: np.ndarray = None  # Indices of active transitions in full space

    def diagonalize_nontammdancoff(self, bse, deps_max=None):
        df_S = self.df_S
        H_sS = self.H_sS
        comm = bse.context.comm

        # Check if matrix is already reduced (from reduced basis calculation)
        if self.active_S is not None:
            # Matrix is already reduced - no filtering needed
            bse.context.print('  Using numpy.linalg.eig (pre-reduced matrix)...')
            nR = len(self.deps_S)
            exclude_S = np.array([], dtype=int)  # Nothing to exclude
            bse.context.print('  Matrix size: %s (pre-filtered)' % nR)

            # Collect and diagonalize
            H_RR = bse.collect_A_SS(H_sS)
            w_T = np.zeros(nR, complex)
            v_RT = None
            if comm.rank == 0:
                w_T, v_RT = np.linalg.eig(H_RR)
            comm.broadcast(w_T, 0)
            return w_T, v_RT, exclude_S

        # Standard path: filter and diagonalize
        if deps_max is None:
            deps_max = self.deps_max
        else:
            # Convert from eV to Hartree for comparison with deps_S
            deps_max = deps_max / Hartree
        excludef_S = np.where(np.abs(df_S) < 0.001)[0]
        excludedeps_S = np.where(np.abs(self.deps_S) > deps_max)[0]
        exclude_S = np.unique(np.concatenate((excludef_S, excludedeps_S)))
        bse.context.print('  Using numpy.linalg.eig...')
        bse.context.print('  Eliminated %s pair orbitals' % len(
            exclude_S))
        H_SS = bse.collect_A_SS(H_sS)
        w_T = np.zeros(bse.nS - len(exclude_S), complex)
        v_ST = None
        if comm.rank == 0:
            H_SS = np.delete(H_SS, exclude_S, axis=0)
            H_SS = np.delete(H_SS, exclude_S, axis=1)
            w_T, v_ST = np.linalg.eig(H_SS)
        else:
            v_ST = None
        comm.broadcast(w_T, 0)
        return w_T, v_ST, exclude_S

    def diagonalize_tammdancoff(self, bse, deps_max=None, elpa=False):
        comm = bse.context.comm

        # Check if matrix is already reduced (from reduced basis calculation)
        if self.active_S is not None:
            # Matrix is already reduced - no filtering needed
            nR = len(self.deps_S)
            exclude_S = np.array([], dtype=int)  # Nothing to exclude
            bse.context.print('  Matrix size: %s (pre-filtered)' % nR)

            if comm.size == 1:
                bse.context.print('  Using lapack (pre-reduced matrix)...')
                w_T, v_Rt = eigh(self.H_sS)
                return w_T, v_Rt, exclude_S

            # Parallel case: use existing parallel infrastructure
            # The matrix H_sS is already in reduced form (nR x nR)
            grid = BlacsGrid(comm, comm.size, 1)
            nr = -((-nR) // comm.size)
            desc = grid.new_descriptor(nR, nR, nr, nR)

            # Redistribute to proper BLACS layout
            from gpaw.old.matrix import suggest_blocking
            nrows, ncols, blocksize = suggest_blocking(nR, comm.size)
            new_grid = BlacsGrid(comm, nrows, ncols)
            new_desc = new_grid.new_descriptor(nR, nR, blocksize, blocksize)
            H_rr = new_desc.zeros(dtype=complex)
            Redistributor(comm, desc, new_desc).redistribute(self.H_sS, H_rr)

            w_T = np.empty(nR)
            v_rt = new_desc.empty(dtype=complex)
            if elpa:
                bse.context.print('Using elpa (pre-reduced matrix)...')
                elpa_obj = LibElpa(new_desc)
                elpa_obj.diagonalize(H_rr, v_rt, w_T)
            else:
                bse.context.print('Using scalapack (pre-reduced matrix)...')
                new_desc.diagonalize_dc(H_rr, v_rt, w_T)

            # Redistribute eigenvectors
            grid_tR = BlacsGrid(comm, comm.size, 1)
            nt = -((-nR) // comm.size)
            desc_tR = grid_tR.new_descriptor(nR, nR, nt, nR)
            v_tR = desc_tR.zeros(dtype=complex)
            Redistributor(comm, new_desc, desc_tR).redistribute(v_rt, v_tR)
            v_Rt = v_tR.conj().T
            return w_T, v_Rt, exclude_S

        # Standard path: filter and diagonalize
        if deps_max is None:
            deps_max = self.deps_max
        else:
            # Convert from eV to Hartree for comparison with deps_S
            deps_max = deps_max / Hartree
        exclude_S = np.where(np.abs(self.deps_S) > deps_max)[0]
        H_rr, new_grid_desc = self.exclude_states(bse, exclude_S)
        if comm.size == 1:
            bse.context.print('  Using lapack...')
            w_T, v_Rt = eigh(H_rr)
            return w_T, v_Rt, exclude_S
        nR = bse.nS - len(exclude_S)
        w_T = np.empty(nR)
        v_rt = new_grid_desc.empty(dtype=complex)
        if elpa:
            bse.context.print('Using elpa...')
            elpa = LibElpa(new_grid_desc)
            elpa.diagonalize(H_rr, v_rt, w_T)
        else:
            bse.context.print('Using scalapack...')
            new_grid_desc.diagonalize_dc(H_rr, v_rt, w_T)

        # redistribute eigenvectors
        # we want them to be parallelized over the last index only

        grid_tR = BlacsGrid(comm, comm.size, 1)
        nt = -((-nR) // comm.size)
        desc_tR = grid_tR.new_descriptor(nR, nR, nt, nR)
        v_tR = desc_tR.zeros(dtype=complex)
        Redistributor(comm, new_grid_desc, desc_tR).redistribute(v_rt, v_tR)
        v_Rt = v_tR.conj().T
        return w_T, v_Rt, exclude_S

    def exclude_states(self, bse, exclude_S):
        """
        Removes pairs from the BSE Hamiltonian.
        A pair is removed if the absolute value of the
        transition energy eps_c - eps_v is greater than deps_max
        """
        H_sS = self.H_sS
        comm = bse.context.comm
        grid = BlacsGrid(comm, comm.size, 1)
        nS = bse.nS
        ns = bse.ns
        desc = grid.new_descriptor(nS, nS, ns, nS)
        H_rr, new_desc = parallel_delete(H_sS, exclude_S, desc)
        bse.context.print('  Eliminated %s pair orbitals' % len(
            exclude_S))
        return H_rr, new_desc


def parallel_delete(A_nn: np.ndarray,
                    deleteN: np.ndarray,
                    grid_desc: BlacsDescriptor,
                    new_desc=None):
    """
    Removes rows and columns from the distributed square matrix A_nn.
    This is done by redistributing the matrix to first make the second index
    global (A_nn -> A_mN), and then deleting along the second (global) index;
    then repeating the same procedure for the first index.
    --------------
    Parameters:
    A_nn: distributed matrix
    deleteN : list of (global) indices to delete
    grid_desc: BlacsDescriptor for A_nn
    new_grid: BlacsGrid on which A_nn will be returned; optional.
    If None, the output grid will be determined automatically
    by the gpaw.old.matrix.suggest_blocking function.
    -------------------
    Returns:
    A_rs: np.ndarray
    new_desc: BlacsDescriptor

    A_rs is the (new) distributed matrix after rows and columns
    have been deleted.
    new_grid is the grid on which it is distributed.
    """
    from gpaw.old.matrix import suggest_blocking
    N = grid_desc.N
    assert N == grid_desc.M, 'Matrix must be square'

    R = N - len(deleteN)  # global matrix dimension after deletion
    dtype = A_nn.dtype
    # redistribute matrix, so it is distributed over first index only.
    # then we can safely delete entries from the second index.
    comm = grid_desc.blacsgrid.comm
    grid_mN = BlacsGrid(comm, comm.size, 1)
    m = -((-N) // comm.size)
    desc_mN = grid_mN.new_descriptor(N, N, m, N)
    A_mN = desc_mN.zeros(dtype=dtype)
    Redistributor(comm, grid_desc, desc_mN).redistribute(A_nn, A_mN)

    # delete, and ensure that array is still contiguous in memory
    A_mR = np.delete(A_mN, deleteN, axis=1)
    A_mR = np.ascontiguousarray(A_mR)
    desc_mR = grid_mN.new_descriptor(N, R, m, R)

    # now distribute over second index, so we can delete entries from 1st
    r = -((-R) // comm.size)
    grid_Nr = BlacsGrid(comm, 1, comm.size)
    desc_Nr = grid_Nr.new_descriptor(N, R, N, r)
    A_Nr = desc_Nr.zeros(dtype=dtype)
    Redistributor(comm, desc_mR, desc_Nr).redistribute(A_mR, A_Nr)
    A_Rr = np.delete(A_Nr, deleteN, axis=0)
    A_Rr = np.ascontiguousarray(A_Rr)
    desc_Rr = grid_Nr.new_descriptor(R, R, R, r)

    # Redistribute to final grid.
    # If this is not specified by the user, we try to find the most
    # efficient grid using the suggest_blocking function.
    if new_desc is None:
        nrows, ncols, blocksize = suggest_blocking(R, comm.size)
        new_grid = BlacsGrid(comm, nrows, ncols)
        new_desc = new_grid.new_descriptor(R, R, blocksize, blocksize)
    A_rr = new_desc.zeros(dtype=dtype)
    Redistributor(comm, desc_Rr, new_desc).redistribute(A_Rr, A_rr)

    return A_rr, new_desc


@dataclass
class ScreenedPotential:
    pawcorr_q: list
    W_qGG: list
    qpd_q: list


class SpinorData:
    def __init__(self, con_m, val_m, e_km, f_km, v_kmn, soc_tol):
        self.e_km = e_km
        self.f_km = f_km
        self.v0_kmn = v_kmn[:, :, ::2]
        self.v1_kmn = v_kmn[:, :, 1::2]

        mi = val_m[0]
        mf = con_m[-1] + 1
        tmp_n = np.argwhere(np.abs(v_kmn[:, mi:mf])**2 > soc_tol)[:, 2]
        self.ni = np.min(tmp_n) // 2
        self.nf = np.max(tmp_n) // 2 + 1

        self.vslice_m = slice(val_m[0], val_m[-1] + 1)
        self.cslice_m = slice(con_m[0], con_m[-1] + 1)

    def _transform_rho(self, K1, K2, slice1, slice2,
                       rho0_nnG, rho1_nnG, susc_component='00'):
        slice_n = slice(self.ni, self.nf)
        vec0k1_mn = self.v0_kmn[K1, slice1, slice_n]
        vec0k2_mn = self.v0_kmn[K2, slice2, slice_n]
        vec1k1_mn = self.v1_kmn[K1, slice1, slice_n]
        vec1k2_mn = self.v1_kmn[K2, slice2, slice_n]
        if susc_component == '00':
            if rho1_nnG is None:
                rho1_nnG = rho0_nnG
            rho0_mmG = np.dot(vec0k1_mn.conj(), np.dot(vec0k2_mn, rho0_nnG))
            rho1_mmG = np.dot(vec1k1_mn.conj(), np.dot(vec1k2_mn, rho1_nnG))
            return rho0_mmG + rho1_mmG
        elif susc_component == '+-':
            assert rho1_nnG is None
            return np.dot(vec0k1_mn.conj(), np.dot(vec1k2_mn, rho0_nnG))
        elif susc_component == '-+':
            assert rho1_nnG is None
            return np.dot(vec1k1_mn.conj(), np.dot(vec0k2_mn, rho0_nnG))
        else:
            raise NotImplementedError('Susceptibility component not '
                                      'implemented. Please choose between '
                                      '"00", "+-" or "-+"')

    def rho_valence_valence(self, K1, K2, rho0_nnG, rho1_nnG=None):
        return self._transform_rho(K1, K2, self.vslice_m,
                                   self.vslice_m, rho0_nnG, rho1_nnG)

    def rho_conduction_conduction(self, K1, K2, rho0_nnG, rho1_nnG=None):
        return self._transform_rho(K1, K2, self.cslice_m, self.cslice_m,
                                   rho0_nnG, rho1_nnG)

    def rho_valence_conduction(self, K1, K2, rho0_nnG,
                               rho1_nnG=None, susc_component='00'):
        return self._transform_rho(K1, K2, self.vslice_m, self.cslice_m,
                                   rho0_nnG, rho1_nnG,
                                   susc_component=susc_component)

    def get_deps(self, K1, K2):
        epsv_m = self.e_km[K1, self.vslice_m]
        epsc_m = self.e_km[K2, self.cslice_m]
        return -(epsv_m[:, np.newaxis] - epsc_m)

    def get_df(self, K1, K2):
        fv_m = self.f_km[K1, self.vslice_m]
        fc_m = self.f_km[K2, self.cslice_m]
        return fv_m[:, np.newaxis] - fc_m


class BSEBackend:
    def __init__(self, *, gs, context,
                 valence_bands,
                 conduction_bands,
                 deps_max=None,
                 add_soc=False,
                 soc_tol=0.0001,
                 ecut=10.,
                 scale=1.0,
                 nbands=None,
                 eshift=None,
                 gw_kn=None,
                 truncation=None,
                 integrate_gamma='reciprocal',
                 q0_correction=False,
                 mode='BSE',
                 q_c=(0.0, 0.0, 0.0),
                 direction=0):

        integrate_gamma = GammaIntegrationMode(integrate_gamma)

        self.gs = gs
        self.q_c = list(q_c)
        self.direction = direction
        self.context = context
        self.add_soc = add_soc
        self.scale = scale
        self.q0_correction = q0_correction
        if q0_correction and truncation != '2D':
            raise ValueError('q0_correction should only be used with '
                             'truncation=\'2D\'.')
        assert mode in ['RPA', 'BSE']

        if deps_max is None:
            self.deps_max = np.inf
        else:
            self.deps_max = deps_max / Hartree
        self.ecut = ecut / Hartree
        self.nbands = nbands
        self.mode = mode

        if integrate_gamma.is_analytical and truncation is not None:
            self.context.print('***WARNING*** Analytical Coulomb integration' +
                               ' is not expected to work with Coulomb ' +
                               'truncation. ' +
                               'Use integrate_gamma=\'reciprocal\'')
        self.integrate_gamma = integrate_gamma

        # Find q-vectors and weights in the IBZ:
        self.kd = self.gs.kd
        if -1 in self.kd.bz2bz_ks:
            self.context.print('***WARNING*** Symmetries may not be right. '
                               'Use gamma-centered grid to be sure')
        offset_c = 0.5 * ((self.kd.N_c + 1) % 2) / self.kd.N_c
        bzq_qc = monkhorst_pack(self.kd.N_c) + offset_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.gs.atoms, self.kd.symmetry)

        # By default calculate the density-density response
        self.susc_component = '00'

        # Bands and spin
        self.nspins = self.gs.nspins
        self.val_m = self.parse_bands(valence_bands, gs=self.gs,
                                      band_type='valence',
                                      add_soc=self.add_soc)
        self.con_m = self.parse_bands(conduction_bands, gs=self.gs,
                                      band_type='conduction',
                                      add_soc=self.add_soc)

        self.use_tammdancoff = decide_whether_tammdancoff(self.val_m,
                                                          self.con_m)

        self.nK = self.kd.nbzkpts
        self.nv = len(self.val_m)
        self.nc = len(self.con_m)
        if eshift is not None:
            eshift /= Hartree
        if gw_kn is not None:
            assert self.nv + self.nc == len(gw_kn[0])
            assert self.kd.nibzkpts == len(gw_kn)
            gw_kn = gw_kn[self.kd.bz2ibz_k]
            gw_kn /= Hartree
        self.gw_kn = gw_kn
        self.eshift = eshift

        self.coulomb = CoulombKernel.from_gs(self.gs, truncation=truncation)

        # Distribution of kpoints
        self.myKrange, self.myKsize = self.parallelisation_kpoints()

        # Number of global and local pair orbitals. Note that self.ns is the
        # the same everywhere and adds up to a value that is larger that nS.
        # This is required for BlacsGrids in the ScalaPack diagonalization.
        self.nS = self.nK * self.nv * self.nc
        comm = self.context.comm
        self.ns = -(-self.nK // comm.size) * self.nv * self.nc

        # Print all the details
        self.print_initialization(self.use_tammdancoff, self.eshift,
                                  self.gw_kn)

        # Treat spin-polarized states as spinors without soc
        if self.nspins == 2 and not self.add_soc:
            self.add_soc = True
            self.scale = 0.0

        # Setup bands
        if self.add_soc:
            self.spinors_data = self._spinordata(soc_tol)
            # Get a wide range of pair densities to allow for SOC mixing.
            # The number of no-SOC states included are determined by soc_tol
            # such that components of the soc eigenstates are included if
            # their norm square is above soc_tol.
            # The no-SOC pair densities are then transformed to the SOC pair
            # densities. vi, vf, ci, cf here determines initial and final
            # indices of no-SOC valence ond conduction states included.
            self.vi = self.spinors_data.ni
            self.vf = self.spinors_data.nf
            self.ci = self.spinors_data.ni
            self.cf = self.spinors_data.nf
        else:
            # Here we just need the pair densities of the specified bands
            self.vi, self.vf = self.val_m[0], self.val_m[-1] + 1
            self.ci, self.cf = self.con_m[0], self.con_m[-1] + 1

    @staticmethod
    def parse_bands(bands, gs, band_type, add_soc):
        """Helper function that checks whether bands are correctly specified,
         and brings them to the format used later in the code.

        Either integers (numbers of desired bands) or lists of band indices
        must be provided. For spin-polarized calculations all
        valence/condiction bands must be specified as a single list (one
        for each) - regardless of spin. Same as if one includes SOC.

        band_type is an optional parameter that is relevant when a desired
        number of bands is given (rather than a list) to help figure out the
        correct band indices.
        """
        if hasattr(bands, '__iter__'):
            if not isinstance(bands[0], int):
                raise ValueError('The bands must be specified as a single '
                                 'list or an integer (number of bands).')
            return bands

        if bands <= 0:
            raise ValueError(
                f'\'bands\' must be a positive integer (received {bands=}).')
        n_fully_occupied_bands, n_partially_occupied_bands = \
            gs.count_occupied_bands()

        if gs.nspins == 2:
            n_fully_occupied_bands += n_partially_occupied_bands
        elif add_soc:
            n_fully_occupied_bands *= 2

        if band_type == 'valence':
            if bands > n_fully_occupied_bands:
                raise ValueError(
                    f'{bands} valence bands were requested, '
                    f'but at most {n_fully_occupied_bands} are available.')
            bands_m = range(n_fully_occupied_bands - bands,
                            n_fully_occupied_bands)

        elif band_type == 'conduction':
            bands_m = range(n_fully_occupied_bands,
                            n_fully_occupied_bands + bands)
        else:
            raise ValueError(f'Invalid band type: {band_type}')

        return bands_m

    def _spinordata(self, soc_tol):
        self.context.print('Diagonalizing spin-orbit Hamiltonian')
        # We dont need ALL the screening bands to mix in the SOC here.
        # This will give us bands up to two times the highest conduction band.
        n2 = np.min([self.con_m[-1], self.nbands])
        soc = self.gs.soc_eigenstates(n2=n2, scale=self.scale)
        f_km = np.array([wf.f_m for wf in soc])
        e_km = soc.eigenvalues()
        e_km /= Hartree
        v_kmn = soc.eigenvectors()
        return SpinorData(self.con_m, self.val_m, e_km, f_km, v_kmn, soc_tol)

    def _precompute_transition_energies(self, kptpair_factory):
        """Pre-compute transition energies for all k-points to enable early filtering.

        Returns:
            deps_Kmm: Transition energies for all K, valence, conduction combinations
            active_Kmm: Boolean mask of active transitions (|deps| <= deps_max)
            n_active: Total number of active transitions
        """
        deps_Kmm = np.zeros((self.nK, self.nv, self.nc), float)

        get_pair = kptpair_factory.get_kpoint_pair
        qpd0 = SingleQPWDescriptor.from_q(self.q_c, self.ecut, self.gs.gd)

        for ik, iK in enumerate(self.myKrange):
            iKq = self.gs.kd.find_k_plus_q(self.q_c, [iK])[0]

            if self.gw_kn is not None:
                epsv_m = self.gw_kn[iK, :self.nv]
                epsc_m = self.gw_kn[iKq, self.nv:]
                deps_Kmm[iK] = -(epsv_m[:, np.newaxis] - epsc_m)
            elif self.add_soc:
                deps_Kmm[iK] = self.spinors_data.get_deps(iK, iKq)
            else:
                pair0 = get_pair(qpd0, 0, iK, self.vi, self.vf,
                                 self.ci, self.cf)
                deps_Kmm[iK] = -pair0.get_transition_energies()

        # Gather transition energies from all ranks
        self.context.comm.sum(deps_Kmm)

        # Apply scissors operator if specified
        if self.eshift is not None:
            # Need occupation factors to apply scissors correctly
            # For now, apply uniformly (conservative estimate)
            deps_Kmm[deps_Kmm > 0] += self.eshift
            deps_Kmm[deps_Kmm < 0] -= self.eshift

        # Create active transition mask based on deps_max
        active_Kmm = np.abs(deps_Kmm) <= self.deps_max
        n_active = np.sum(active_Kmm)

        return deps_Kmm, active_Kmm, n_active

    def _compute_reduced_index_mappings(self, active_Kmm):
        """Compute mappings between full (K,v,c) indices and reduced indices.

        Returns:
            S_to_R: array of shape (nS,) mapping full index S to reduced R,
                    or -1 if transition S is not active
            R_to_S: array of shape (nR,) mapping reduced index R to full S
            R_to_Kvc: array of shape (nR, 3) mapping reduced R to (K, v, c)
            my_r_to_R: array mapping local reduced indices to global R
            my_R_range: tuple (R_start, R_end) for this rank's reduced indices
        """
        nK, nv, nc = self.nK, self.nv, self.nc
        nS = nK * nv * nc

        # Flatten active_Kmm to active_S
        active_S = active_Kmm.ravel()
        nR = np.sum(active_S)

        # Create S -> R mapping
        S_to_R = np.full(nS, -1, dtype=int)
        S_to_R[active_S] = np.arange(nR)

        # Create R -> S mapping
        R_to_S = np.where(active_S)[0]

        # Create R -> (K, v, c) mapping
        R_to_Kvc = np.zeros((nR, 3), dtype=int)
        for R, S in enumerate(R_to_S):
            K = S // (nv * nc)
            vc = S % (nv * nc)
            v = vc // nc
            c = vc % nc
            R_to_Kvc[R] = [K, v, c]

        # Compute local reduced indices for this rank
        # Each rank owns transitions for k-points in myKrange
        my_active_count = 0
        my_R_start = 0
        for iK in range(self.nK):
            count_at_K = np.sum(active_Kmm[iK])
            if iK < self.myKrange[0] if len(self.myKrange) > 0 else 0:
                my_R_start += count_at_K
            if iK in self.myKrange:
                my_active_count += count_at_K

        my_R_range = (my_R_start, my_R_start + my_active_count)

        # Create mapping from local r index to global R
        my_r_to_R = np.arange(my_R_start, my_R_start + my_active_count)

        return S_to_R, R_to_S, R_to_Kvc, my_r_to_R, my_R_range, nR

    @timer('BSE calculate')
    def calculate(self, optical, irreducible=False):
        """Calculate the BSE Hamiltonian. This includes setting up all
        machinery for pair densities, KS eignevalues and occupation factors.
        At the end the direct and indirect interaction are included through
        calls to separate functions.

        The indices are indicative such that all capital letters imply
        global indices and lower case letters imply local ("my CPU")
        indices. For example, K, S and k, s are global and local k-point
        and pair state indices respectively.

        In addition the KS state indices are used such that n represents
        states without SOC and m represents states with SOC. This is not always
        possible though - the Hamiltonian, for example, is always
        denoted H_kmmKmm - also for calculations without SOC. G is reciprocal
        lattice index.

        The parameter 'irreducible' puts V=0 such that the BSE kernel only
        contians W. It is used for BSE+ calculations.
        """
        qpd0 = SingleQPWDescriptor.from_q(self.q_c, self.ecut, self.gs.gd)

        self.ikq_k = self.kd.find_k_plus_q(self.q_c)
        if irreducible:
            self.v_G = np.zeros(qpd0.ng_q[0])
        else:
            self.v_G = self.coulomb.V(qpd=qpd0, q_v=None)

        if optical:
            self.v_G[0] = 0.0

        context = ResponseContext(txt='pair.txt', timer=self.context.timer,
                                  comm=serial_comm)
        kptpair_factory = KPointPairFactory(gs=self.gs, context=context)
        pair_calc = kptpair_factory.pair_calculator()
        pawcorr = self.gs.pair_density_paw_corrections(qpd0)

        if self.mode != 'RPA':
            screened_potential = self.calculate_screened_potential()
        else:
            screened_potential = None

        # Early filtering optimization: pre-compute transition energies
        # to identify which transitions will be filtered out by deps_max.
        # This allows skipping expensive pair density calculations for
        # transitions that will be discarded anyway.
        use_early_filter = np.isfinite(self.deps_max)
        if use_early_filter:
            self.context.timer.start('Pre-compute transition energies')
            deps_Kmm_precomputed, active_Kmm, n_active = \
                self._precompute_transition_energies(kptpair_factory)
            n_total = self.nK * self.nv * self.nc
            n_filtered = n_total - n_active
            self.context.print(
                f'Early filtering: {n_filtered}/{n_total} transitions '
                f'({100*n_filtered/n_total:.1f}%) will be skipped '
                f'(|deps| > {self.deps_max * Hartree:.2f} eV)')
            self.context.timer.stop('Pre-compute transition energies')
            # Store for later use in Hamiltonian construction
            self._active_Kmm = active_Kmm
        else:
            self._active_Kmm = None

        # Calculate pair densities, eigenvalues and occupations
        self.context.timer.start('Pair densities')
        if self.susc_component != '00':
            assert self.nspins == 2
            rhomag_KmmG = np.zeros((self.nK, self.nv,
                                    self.nc, len(self.v_G)), complex)
        rhoex_KmmG = np.zeros((self.nK, self.nv,
                               self.nc, len(self.v_G)), complex)
        df_Kmm = np.zeros((self.nK, self.nv,
                           self.nc), float)  # (fc - fv)
        deps_kmm = np.zeros((self.myKsize, self.nv,
                             self.nc), float)  # (ec - ev)
        deps_Kmm = np.zeros((self.nK, self.nv, self.nc), float)
        optical_limit = np.allclose(self.q_c, 0.0)

        get_pair = kptpair_factory.get_kpoint_pair
        get_pair_density = pair_calc.get_pair_density

        # Calculate all properties diagonal in k-point
        # These include the indirect (exchange) kernel,
        # pseudo-energies, and occupation numbers
        for ik, iK in enumerate(self.myKrange):
            # Early filtering: skip k-points with no active transitions
            if use_early_filter and not active_Kmm[iK].any():
                # Still need to store transition energies for completeness
                deps_kmm[ik] = deps_Kmm_precomputed[iK]
                continue
            pair0 = get_pair(qpd0, 0, iK, self.vi, self.vf,
                             self.ci, self.cf)
            v_n = np.arange(self.vi, self.vf)
            c_n = np.arange(self.ci, self.cf)
            iKq = self.gs.kd.find_k_plus_q(self.q_c, [iK])[0]

            # Energies
            if self.gw_kn is not None:
                epsv_m = self.gw_kn[iK, :self.nv]
                epsc_m = self.gw_kn[iKq, self.nv:]
                deps_kmm[ik] = -(epsv_m[:, np.newaxis] - epsc_m)
            elif self.add_soc:
                deps_kmm[ik] = self.spinors_data.get_deps(iK, iKq)
            else:
                deps_kmm[ik] = -pair0.get_transition_energies()
            if optical_limit:
                deps_kmm[np.where(deps_kmm == 0)] = 1.0e-9

            # Occupation factors
            if self.add_soc:
                df_Kmm[iK] = self.spinors_data.get_df(iK, iKq)
            else:
                df_Kmm[iK] = pair0.get_occupation_differences()

            # Pair densities
            rho0_nnG = get_pair_density(qpd0, pair0, v_n, c_n,
                                        pawcorr=pawcorr)
            if optical_limit:
                n0_nnv = pair_calc.get_optical_pair_density_head(
                    qpd0, pair0, v_n, c_n)
                rho0_nnG[:, :, 0] = n0_nnv[:, :, self.direction]
            if self.nspins == 2:
                pair1 = get_pair(qpd0, 1, iK, self.vi, self.vf,
                                 self.ci, self.cf)
                rho1_nnG = get_pair_density(qpd0, pair1, v_n, c_n,
                                            pawcorr=pawcorr)
                if optical_limit:
                    n1_nnv = pair_calc.get_optical_pair_density_head(
                        qpd0, pair1, v_n, c_n)
                    rho1_nnG[:, :, 0] = n1_nnv[:, :, self.direction]
                    deps1_nn = -pair1.get_transition_energies()
                    rho1_nnG[:, :, 0] *= deps1_nn
            else:
                rho1_nnG = None

            # Generate the pair density matrix (with soc) used below
            if self.add_soc:
                if optical_limit:
                    deps0_nn = -pair0.get_transition_energies()
                    rho0_nnG[:, :, 0] *= deps0_nn
                rhoex_KmmG[iK] = \
                    self.spinors_data.rho_valence_conduction(
                        iK, iKq, rho0_nnG, rho1_nnG)
                if optical_limit:
                    rhoex_KmmG[iK, :, :, 0] /= deps_kmm[ik]
            else:
                rhoex_KmmG[iK] = rho0_nnG

            # Generate the magnetic spin flip pair density for magnons
            if self.susc_component != '00':
                if self.susc_component == '+-':
                    s = 0
                elif self.susc_component == '-+':
                    s = 1
                pairflip = get_pair(qpd0, s, iK, self.vi, self.vf,
                                    self.ci, self.cf, flipspin=True)
                rhoflip_nnG = get_pair_density(qpd0, pairflip, v_n, c_n,
                                               pawcorr=pawcorr)
                rhomag_KmmG[iK] = self.spinors_data.rho_valence_conduction(
                    iK, iKq, rhoflip_nnG, susc_component=self.susc_component)

        # Scissors operator shift
        if self.eshift is not None:
            deps_kmm[np.where(df_Kmm[self.myKrange] > 1e-3)] += self.eshift
            deps_kmm[np.where(df_Kmm[self.myKrange] < -1e-3)] -= self.eshift
        deps_Kmm[self.myKrange] = deps_kmm

        comm = self.context.comm
        comm.sum(deps_Kmm)
        comm.sum(df_Kmm)
        comm.sum(rhoex_KmmG)

        self.rhoG0_S = np.reshape(rhoex_KmmG[:, :, :, 0], -1)
        self.rho_SG = np.reshape(rhoex_KmmG, (len(self.rhoG0_S), -1))
        if self.susc_component != '00':
            comm.sum(rhomag_KmmG)
            self.rhomag_SG = np.reshape(rhomag_KmmG, (self.nS, -1))
            G_Gv = qpd0.get_reciprocal_vectors(add_q=False)
            self.G_Gc = np.dot(G_Gv, qpd0.gd.cell_cv.T / (2 * np.pi))
        self.context.timer.stop('Pair densities')

        # Calculate Hamiltonian
        self.context.timer.start('Calculate Hamiltonian')
        t0 = time()

        def update_progress(iK1):
            dt = time() - t0
            tleft = dt * self.myKsize / (iK1 + 1) - dt

            self.context.print(
                '  Finished %s pair orbitals in %s - Estimated %s left'
                % ((iK1 + 1) * self.nv * self.nc * comm.size,
                    timedelta(seconds=round(dt)),
                    timedelta(seconds=round(tleft))))

        self.context.print('Calculating {} matrix elements at q_c = {}'.format(
            self.mode, self.q_c))

        # multiply by 2 when spin-paired and no SOC
        df_Kmm *= 2.0 / self.nK / (self.add_soc + 1)

        # Use reduced basis approach when early filtering is active
        if use_early_filter:
            # Also filter by occupation factor
            df_active = np.abs(df_Kmm) >= 0.001
            active_Kmm = active_Kmm & df_active

            # Recompute filtering statistics
            n_active = np.sum(active_Kmm)
            n_total = self.nK * self.nv * self.nc
            n_filtered = n_total - n_active

            # Compute reduced index mappings
            S_to_R, R_to_S, R_to_Kvc, my_r_to_R, my_R_range, nR = \
                self._compute_reduced_index_mappings(active_Kmm)
            self._my_R_range = my_R_range
            my_nR = my_R_range[1] - my_R_range[0]

            self.context.print(
                f'Reduced basis: {n_active}/{n_total} transitions '
                f'({100*n_active/n_total:.1f}%) kept after deps_max and df filtering')
            self.context.print(
                f'Memory savings: {100*(1 - n_active**2/n_total**2):.1f}% '
                f'(reduced matrix size: {nR}x{nR} vs {n_total}x{n_total})')

            # Build reduced pair density array
            nG = rhoex_KmmG.shape[-1]
            rhoex_RG = np.zeros((nR, nG), complex)
            for R in range(nR):
                K, v, c = R_to_Kvc[R]
                rhoex_RG[R] = rhoex_KmmG[K, v, c]

            # Store reduced versions for spectral weights
            self.rhoG0_R = rhoex_RG[:, 0]
            self.rho_RG = rhoex_RG
            self._R_to_S = R_to_S  # For mapping back to full space

            # Allocate reduced Hamiltonian
            H_rR = np.zeros((my_nR, nR), complex)

            # Add kernels in reduced basis
            self.add_indirect_kernel_reduced(rhoex_RG, H_rR)
            if self.mode != 'RPA':
                self.add_direct_kernel_reduced(
                    kptpair_factory, pair_calc, screened_potential,
                    update_progress, H_rR, active_Kmm, S_to_R)

            H_rR /= self.gs.volume
            self.context.timer.stop('Calculate Hamiltonian')

            # Build reduced occupation factors and transition energies
            df_R = df_Kmm.ravel()[R_to_S]
            deps_R = deps_Kmm.ravel()[R_to_S]

            # Apply occupation factors and add diagonal energies
            my_R_start = my_R_range[0]
            for r in range(my_nR):
                R = my_R_start + r
                H_rR[r] *= df_R[R]
                H_rR[r, R] += deps_R[R]

            # Store for BSE methods that need full arrays
            df_S = np.reshape(df_Kmm, -1)
            deps_S = np.reshape(deps_Kmm, -1)
            self.df_S = df_S

            # Return BSEMatrix with reduced Hamiltonian
            # Set deps_max=inf to signal no further filtering needed
            return BSEMatrix(df_R, H_rR, deps_R, np.inf, active_S=R_to_S)

        else:
            # Original full-basis approach
            # Hamiltonian buffer array
            H_kmmKmm = np.zeros((self.myKsize, self.nv, self.nc,
                                 self.nK, self.nv, self.nc),
                                complex)

            # Add kernels to buffer array
            self.add_indirect_kernel(kptpair_factory, rhoex_KmmG, H_kmmKmm)
            if self.mode != 'RPA':
                self.add_direct_kernel(kptpair_factory, pair_calc,
                                       screened_potential, update_progress,
                                       H_kmmKmm)
            H_kmmKmm /= self.gs.volume
            self.context.timer.stop('Calculate Hamiltonian')

            if self.myKsize > 0:
                iS0 = self.myKrange[0] * self.nv * self.nc

            df_S = np.reshape(df_Kmm, -1)
            self.df_S = df_S

            deps_S = np.reshape(deps_Kmm, -1)
            deps_s = np.reshape(deps_kmm, -1)

            mySsize = self.myKsize * self.nv * self.nc
            H_sS = np.reshape(H_kmmKmm, (mySsize, self.nS))
            for iS in range(mySsize):
                # Multiply by occupations
                H_sS[iS] *= df_S[iS0 + iS]
                # add bare transition energies
                H_sS[iS, iS0 + iS] += deps_s[iS]

            return BSEMatrix(df_S, H_sS, deps_S, self.deps_max)

    @timer('add_direct_kernel')
    def add_direct_kernel(self, kptpair_factory, pair_calc, screened_potential,
                          update_progress, H_kmmKmm):
        kpf = kptpair_factory
        active_Kmm = self._active_Kmm  # May be None if no early filtering
        for ik1, iK1 in enumerate(self.myKrange):
            # Early filtering: skip if no active transitions at this k-point
            if active_Kmm is not None and not active_Kmm[iK1].any():
                continue
            kptv1_s = [kpf.get_k_point(s, iK1, self.vi, self.vf)
                       for s in range(self.nspins)]
            kptc1_s = [kpf.get_k_point(s, self.ikq_k[iK1], self.ci, self.cf)
                       for s in range(self.nspins)]
            for Q_c in self.qd.bzk_kc:
                iK2 = self.kd.find_k_plus_q(Q_c, [kptv1_s[0].K])[0]
                # Early filtering: skip if no active transitions at K2
                if active_Kmm is not None and not active_Kmm[iK2].any():
                    continue
                kptv2_s = [kptpair_factory.get_k_point(s, iK2, self.vi,
                                                       self.vf)
                           for s in range(self.nspins)]
                kptc2_s = [kptpair_factory.get_k_point(s, self.ikq_k[iK2],
                                                       self.ci, self.cf)
                           for s in range(self.nspins)]

                rho3_nnG, iq = self.get_density_matrix(
                    pair_calc, screened_potential, kptv1_s[0], kptv2_s[0])

                rho4_nnG, iq = self.get_density_matrix(
                    pair_calc, screened_potential, kptc1_s[0], kptc2_s[0])

                if self.nspins == 2:
                    rho3s1_nnG, iq = self.get_density_matrix(
                        pair_calc, screened_potential, kptv1_s[1], kptv2_s[1])

                    rho4s1_nnG, iq = self.get_density_matrix(
                        pair_calc, screened_potential, kptc1_s[1], kptc2_s[1])
                else:
                    rho3s1_nnG = None
                    rho4s1_nnG = None

                # Here we use n instead of m for the soc indices to save memory
                if self.add_soc:
                    rho3_nnG = self.spinors_data.rho_valence_valence(
                        kptv1_s[0].K, kptv2_s[0].K, rho3_nnG, rho3s1_nnG)

                    rho4_nnG = self.spinors_data.rho_conduction_conduction(
                        kptc1_s[0].K, kptc2_s[0].K, rho4_nnG, rho4s1_nnG)

                self.context.timer.start('Screened exchange')
                W_mmmm = np.einsum(
                    'ijk,km,pqm->ipjq',
                    rho3_nnG.conj(),
                    screened_potential.W_qGG[iq],
                    rho4_nnG,
                    optimize='optimal')
                # Only include 0.5*W for spinpaired calculations without soc
                H_kmmKmm[ik1, :, :, iK2] -= W_mmmm * (self.add_soc + 1) / 2
                self.context.timer.stop('Screened exchange')

            if iK1 % (self.myKsize // 5 + 1) == 0:
                update_progress(iK1=iK1)

    @timer('add_indirect_kernel')
    def add_indirect_kernel(self, kptpair_factory, rhoex_KmmG, H_kmmKmm):
        active_Kmm = self._active_Kmm  # May be None if no early filtering
        for ik1, iK1 in enumerate(self.myKrange):
            # Early filtering: skip if no active transitions at this k-point
            if active_Kmm is not None and not active_Kmm[iK1].any():
                continue
            kptv1 = kptpair_factory.get_k_point(
                0, iK1, self.vi, self.vf)
            rho1V_mmG = rhoex_KmmG.conj()[iK1, :, :] * self.v_G
            for Q_c in self.qd.bzk_kc:
                iK2 = self.kd.find_k_plus_q(Q_c, [kptv1.K])[0]
                # Early filtering: skip if no active transitions at K2
                if active_Kmm is not None and not active_Kmm[iK2].any():
                    continue
                rho2_mmG = rhoex_KmmG[iK2]
                self.context.timer.start('Coulomb')
                H_kmmKmm[ik1, :, :, iK2, :, :] += np.einsum(
                    'ijG,mnG->ijmn', rho1V_mmG, rho2_mmG,
                    optimize='optimal')
                self.context.timer.stop('Coulomb')

    @timer('add_indirect_kernel_reduced')
    def add_indirect_kernel_reduced(self, rhoex_RG, H_rR):
        """Compute indirect (exchange) kernel directly in reduced basis.

        This is more memory-efficient than the full basis when many
        transitions are filtered by deps_max.

        Args:
            rhoex_RG: Pair densities for active transitions only, shape (nR, nG)
            H_rR: Reduced Hamiltonian to add kernel to, shape (my_nR, nR)
        """
        self.context.timer.start('Coulomb reduced')
        # H[r, R] = sum_G rho*[r,G] * v[G] * rho[R,G]
        # For local rows only
        my_R_start, my_R_end = self._my_R_range
        rhoV_rG = rhoex_RG[my_R_start:my_R_end].conj() * self.v_G
        H_rR += np.dot(rhoV_rG, rhoex_RG.T)
        self.context.timer.stop('Coulomb reduced')

    @timer('add_direct_kernel_reduced')
    def add_direct_kernel_reduced(self, kptpair_factory, pair_calc,
                                   screened_potential, update_progress,
                                   H_rR, active_Kmm, S_to_R):
        """Compute direct (screened exchange) kernel with reduced output.

        Computes full blocks for each k-point pair, then extracts only
        the elements corresponding to active transitions.

        Args:
            H_rR: Reduced Hamiltonian to add kernel to, shape (my_nR, nR)
            active_Kmm: Boolean mask of active transitions, shape (nK, nv, nc)
            S_to_R: Mapping from full S index to reduced R index
        """
        kpf = kptpair_factory
        nv, nc = self.nv, self.nc
        my_R_start, my_R_end = self._my_R_range

        # Pre-compute active (v,c) indices for each k-point
        active_vc_K = []
        for iK in range(self.nK):
            active_vc = np.array(np.where(active_Kmm[iK])).T  # shape (n_active, 2)
            active_vc_K.append(active_vc)

        for ik1, iK1 in enumerate(self.myKrange):
            active_vc1 = active_vc_K[iK1]
            if len(active_vc1) == 0:
                continue

            kptv1_s = [kpf.get_k_point(s, iK1, self.vi, self.vf)
                       for s in range(self.nspins)]
            kptc1_s = [kpf.get_k_point(s, self.ikq_k[iK1], self.ci, self.cf)
                       for s in range(self.nspins)]

            for Q_c in self.qd.bzk_kc:
                iK2 = self.kd.find_k_plus_q(Q_c, [kptv1_s[0].K])[0]
                active_vc2 = active_vc_K[iK2]
                if len(active_vc2) == 0:
                    continue

                kptv2_s = [kptpair_factory.get_k_point(s, iK2, self.vi,
                                                       self.vf)
                           for s in range(self.nspins)]
                kptc2_s = [kptpair_factory.get_k_point(s, self.ikq_k[iK2],
                                                       self.ci, self.cf)
                           for s in range(self.nspins)]

                rho3_nnG, iq = self.get_density_matrix(
                    pair_calc, screened_potential, kptv1_s[0], kptv2_s[0])

                rho4_nnG, iq = self.get_density_matrix(
                    pair_calc, screened_potential, kptc1_s[0], kptc2_s[0])

                if self.nspins == 2:
                    rho3s1_nnG, iq = self.get_density_matrix(
                        pair_calc, screened_potential, kptv1_s[1], kptv2_s[1])
                    rho4s1_nnG, iq = self.get_density_matrix(
                        pair_calc, screened_potential, kptc1_s[1], kptc2_s[1])
                else:
                    rho3s1_nnG = None
                    rho4s1_nnG = None

                if self.add_soc:
                    rho3_nnG = self.spinors_data.rho_valence_valence(
                        kptv1_s[0].K, kptv2_s[0].K, rho3_nnG, rho3s1_nnG)
                    rho4_nnG = self.spinors_data.rho_conduction_conduction(
                        kptc1_s[0].K, kptc2_s[0].K, rho4_nnG, rho4s1_nnG)

                self.context.timer.start('Screened exchange')
                W_mmmm = np.einsum(
                    'ijk,km,pqm->ipjq',
                    rho3_nnG.conj(),
                    screened_potential.W_qGG[iq],
                    rho4_nnG,
                    optimize='optimal')

                # Extract active elements and add to reduced Hamiltonian
                factor = -(self.add_soc + 1) / 2
                for v1, c1 in active_vc1:
                    S1 = iK1 * nv * nc + v1 * nc + c1
                    R1 = S_to_R[S1]
                    r1 = R1 - my_R_start  # local index
                    for v2, c2 in active_vc2:
                        S2 = iK2 * nv * nc + v2 * nc + c2
                        R2 = S_to_R[S2]
                        H_rR[r1, R2] += W_mmmm[v1, c1, v2, c2] * factor

                self.context.timer.stop('Screened exchange')

            if iK1 % (self.myKsize // 5 + 1) == 0:
                update_progress(iK1=iK1)

    @timer('get_density_matrix')
    def get_density_matrix(self, pair_calc, screened_potential, kpt1, kpt2):
        self.context.timer.start('Symop')
        from gpaw.response.g0w0 import QSymmetryOp, get_nmG
        symop, iq = QSymmetryOp.get_symop_from_kpair(self.kd, self.qd,
                                                     kpt1, kpt2)
        qpd = screened_potential.qpd_q[iq]
        nG = qpd.ngmax
        pawcorr0 = screened_potential.pawcorr_q[iq]
        pawcorr, I_G = symop.apply_symop_q(qpd, pawcorr0, kpt1, kpt2)
        self.context.timer.stop('Symop')

        rho_nnG = np.zeros((len(kpt1.eps_n), len(kpt2.eps_n), nG), complex)
        for n in range(len(rho_nnG)):
            rho_nnG[n] = get_nmG(kpt1, kpt2, pawcorr, n, qpd, I_G,
                                 pair_calc, timer=self.context.timer)

        return rho_nnG, iq

    @cached_property
    def _chi0calc(self):
        return Chi0Calculator(
            self.gs, self.context.with_txt('chi0.txt'),
            wd=FrequencyDescriptor([0.0]),
            eta=0.001,
            ecut=self.ecut * Hartree,
            intraband=False,
            hilbert=False,
            nbands=self.nbands)

    @cached_property
    def blockcomm(self):
        return self._chi0calc.chi0_body_calc.blockcomm

    @cached_property
    def wcontext(self):
        # XXX This was world but I changed it to self.context.comm.
        # Why was it world??  --askhl
        return ResponseContext(txt='w.txt', comm=self.context.comm)

    @cached_property
    def _wcalc(self):
        return initialize_w_calculator(
            self._chi0calc, self.wcontext,
            coulomb=self.coulomb,
            integrate_gamma=self.integrate_gamma,
            q0_correction=self.q0_correction)

    @timer('calculate_screened_potential')
    def calculate_screened_potential(self):
        """Calculate W_GG(q)."""

        pawcorr_q = []
        W_qGG = []
        qpd_q = []

        t0 = time()
        self.context.print('Calculating screened potential')
        for iq, q_c in enumerate(self.qd.ibzk_kc):
            chi0 = self._chi0calc.calculate(q_c)
            W_wGG = self._wcalc.calculate_W_wGG(chi0)
            W_GG = W_wGG[0]
            # This is such a terrible way to access the paw
            # corrections. Attributes should not be groped like
            # this... Change in the future! XXX
            pawcorr_q.append(self._chi0calc.chi0_body_calc.pawcorr)
            qpd_q.append(chi0.qpd)
            W_qGG.append(W_GG)

            if iq % (self.qd.nibzkpts // 5 + 1) == 2:
                dt = time() - t0
                tleft = dt * self.qd.nibzkpts / (iq + 1) - dt
                self.context.print(
                    '  Finished {} q-points in {} - Estimated {} left'.format(
                        iq + 1, timedelta(seconds=round(dt)), timedelta(
                            seconds=round(tleft))))

        return ScreenedPotential(pawcorr_q, W_qGG, qpd_q)

    @timer('diagonalize')
    def diagonalize_bse_matrix(self, bsematrix):
        self.context.print('Diagonalizing Hamiltonian')
        if self.use_tammdancoff:
            return bsematrix.diagonalize_tammdancoff(self)
        else:
            return bsematrix.diagonalize_nontammdancoff(self)

    @timer('get_bse_matrix')
    def get_bse_matrix(self, optical=True, irreducible=False):
        """Calculate BSE matrix."""
        return self.calculate(optical=optical, irreducible=irreducible)

    @timer('get_spectral_weights')
    def get_spectral_weights(self, eig_data, df_S, mode_c):
        if mode_c is None:
            rho_S = self.rhoG0_S
        else:
            G_Gc = self.G_Gc
            index = np.where(np.all(np.round(G_Gc) == mode_c, axis=1))[0][0]
            rho_S = self.rhomag_SG[:, index]

        w_T, v_St = eig_data[0], eig_data[1]
        exclude_S = eig_data[2]
        comm = self.context.comm
        nS = self.nS - len(exclude_S)
        ns = -(-nS // comm.size)
        dft_S = np.delete(df_S, exclude_S)
        rhot_S = np.delete(rho_S, exclude_S)
        C_T = np.zeros(nS, complex)
        # Calculate the spectral weights C_T
        if self.use_tammdancoff:
            A_t = np.dot(rhot_S, v_St)
            B_t = np.dot(rhot_S * dft_S, v_St)
            if comm.size == 1:
                C_T = B_t.conj() * A_t
            else:
                grid = BlacsGrid(comm, comm.size, 1)
                desc = grid.new_descriptor(nS, 1, ns, 1)
                C_t = desc.empty(dtype=complex)
                C_t[:, 0] = B_t.conj() * A_t
                C_T = desc.collect_on_master(C_t)[:, 0]
                if comm.rank != 0:
                    C_T = np.empty(nS, dtype=complex)
                comm.broadcast(C_T, 0)
        else:
            if comm.rank == 0:
                A_T = np.dot(rhot_S, v_St)
                B_T = np.dot(rhot_S * dft_S, v_St)
                tmp = np.dot(v_St.conj().T, v_St)
                overlap_TT = np.linalg.inv(tmp)
                C_T = np.dot(B_T.conj(), overlap_TT.T) * A_T
            comm.broadcast(C_T, 0)

        return w_T, C_T

    def _cache_eig_data(self, irreducible, optical, w_w):
        if (not hasattr(self, 'eig_data')
            or self.eig_data_irreducible != irreducible
            or self.eig_data_optical != optical):
            bsematrix = self.get_bse_matrix(optical=optical,
                                            irreducible=irreducible)
            self.context.print('Calculating response function at %s frequency '
                               'points' % len(w_w))
            self.eig_data = self.diagonalize_bse_matrix(bsematrix)
            self.eig_data_irreducible = irreducible
            self.eig_data_optical = optical

    @timer('get_vchi')
    def get_vchi(self, w_w=None, eta=0.1, optical=True, write_eig=None,
                 mode_c=None, irreducible=False):
        """Returns v * chi where v is the bare Coulomb interaction"""

        vchi_w = np.zeros(len(w_w), dtype=complex)

        self._cache_eig_data(irreducible, optical, w_w)

        w_T, C_T = self.get_spectral_weights(self.eig_data,
                                             self.df_S, mode_c)

        if write_eig is not None:
            assert isinstance(write_eig, str)
            filename = write_eig
            if self.context.comm.rank == 0:
                write_bse_eigenvalues(filename, self.mode,
                                      w_T * Hartree, C_T)

        eta /= Hartree
        for iw, w in enumerate(w_w / Hartree):
            tmp_T = 1. / (w - w_T + 1j * eta)
            vchi_w[iw] += np.dot(tmp_T, C_T)
        vchi_w *= 4 * np.pi / self.gs.volume

        if not np.allclose(self.q_c, 0.0):
            cell_cv = self.gs.gd.cell_cv
            B_cv = 2 * np.pi * np.linalg.inv(cell_cv).T
            q_v = np.dot(self.q_c, B_cv)
            vchi_w /= np.dot(q_v, q_v)

        # Check f-sum rule
        nv = self.gs.nvalence
        dw_w = (w_w[1:] - w_w[:-1]) / Hartree
        wvchi_w = (w_w[1:] * vchi_w[1:] + w_w[:-1] * vchi_w[:-1]) / Hartree / 2
        N = -np.dot(dw_w, wvchi_w.imag) * self.gs.volume / (2 * np.pi**2)
        Nt = 2 * np.dot(w_T, C_T).real
        self.context.print('', flush=False)
        self.context.print('Checking f-sum rule', flush=False)
        self.context.print(f'  Valence electrons : {nv}', flush=False)
        self.context.print(f'  Frequency integral: {N:f}', flush=False)
        self.context.print(f'  Sum of weights    : {Nt:f}', flush=False)
        self.context.print('')

        return vchi_w

    @timer('get_chi_wGG')
    def get_chi_wGG(self, w_w=None, eta=0.1, readfile=None, optical=True,
                    irreducible=False):
        """Returns chi_wGG'"""

        self._cache_eig_data(irreducible, optical, w_w)

        w_T, v_Rt, exclude_S = \
            self.eig_data[0], self.eig_data[1], self.eig_data[2]
        rho_SG = self.rho_SG
        df_S = self.df_S
        df_R = np.delete(df_S, exclude_S)
        rho_RG = np.delete(rho_SG, exclude_S, axis=0)

        comm = self.context.comm
        nG = rho_RG.shape[-1]
        nR = self.nS - len(exclude_S)
        nr = -(-nR // comm.size)
        # nr is the local size of the array

        self.context.print('Calculating response function at %s frequency '
                           'points' % len(w_w))
        self.blocks = Blocks1D(comm, len(w_T))
        w_t = w_T[self.blocks.myslice]

        if not self.use_tammdancoff:
            if comm.rank == 0:
                v_RT = v_Rt
                A_GT = rho_RG.T @ v_RT
                B_GT = rho_RG.T * df_R[np.newaxis] @ v_RT
                tmp = v_RT.conj().T @ v_RT
                overlap_tt = np.linalg.inv(tmp)
                C_tGG = ((B_GT.conj() @ overlap_tt.T).T)[..., np.newaxis] *\
                    A_GT.T[:, np.newaxis]
                C_tGG = C_tGG[:nR].reshape((nR, nG, nG))
                flat_C_tGG = C_tGG.ravel()
            else:
                flat_C_tGG = np.empty(nR * nG * nG, dtype=complex)
            comm.broadcast(flat_C_tGG, 0)
            C_tGG = flat_C_tGG.reshape((nR, nG, nG))[self.blocks.myslice]
            C_tGG1 = None
        else:
            A_Gt = rho_RG.T @ v_Rt
            B_Gt = (rho_RG.T * df_R[np.newaxis]) @ v_Rt
            '''The following computes
               C_tGG1 = A_Gt.T.conj()[..., np.newaxis] * B_Gt.T[:, np.newaxis]
               C_tGG = B_Gt.T.conj()[..., np.newaxis] * A_Gt.T[:, np.newaxis]
               '''
            grid = BlacsGrid(comm, comm.size, 1)
            desc = grid.new_descriptor(nR, nG * nG, nr, nG * nG)
            C_tGG = desc.empty(dtype=complex)
            np.einsum('Gt,Ht->tGH', B_Gt.conj(), A_Gt,
                      out=C_tGG.reshape((-1, nG, nG)))
            desc1 = grid.new_descriptor(nR, nG * nG, nr, nG * nG)
            C_tGG1 = desc1.empty(dtype=complex)
            np.einsum('Gt,Ht->tGH', A_Gt.conj(), B_Gt,
                      out=C_tGG1.reshape((-1, nG, nG)))
            print(f'shape is {C_tGG.shape}')
            C_tGG = C_tGG[:C_tGG.shape[0]].reshape((C_tGG.shape[0], nG, nG))
            C_tGG1 = C_tGG1[:C_tGG1.shape[0]].reshape(
                (C_tGG1.shape[0], nG, nG))

        eta /= Hartree

        if C_tGG is not None:
            tmp_tw = 1 / (w_w[None, :] / Hartree - w_t[:, None] + 1j * eta)
            chi_wGG_local = np.einsum('tw,tAB->wAB', tmp_tw, C_tGG)

            if C_tGG1 is not None:
                n_tmp_tw = - 1 / (w_w[None, :] / Hartree
                                  + w_t[:, None] + 1j * eta)
                chi_wGG_local += np.einsum('tw,tAB->wAB', n_tmp_tw, C_tGG1)

            chi_wGG_local *= 1 / self.gs.volume

        comm.sum(chi_wGG_local)
        chi_wGG = chi_wGG_local

        return np.swapaxes(chi_wGG, -1, -2)

    def get_dielectric_function(self, *args, filename='df_bse.csv', **kwargs):
        vchi = self.vchi(*args, optical=True, **kwargs)
        return vchi.dielectric_function(filename=filename)

    def get_eels_spectrum(self, *args, filename='df_bse.csv', **kwargs):
        vchi = self.vchi(*args, optical=False, **kwargs)
        return vchi.eels_spectrum(filename=filename)

    def get_polarizability(self, *args, filename='pol_bse.csv', **kwargs):
        vchi = self.vchi(*args, optical=True, **kwargs)
        return vchi.polarizability(filename=filename)

    def get_magnetic_susceptibility(self, *args, modes_Gc=[[0, 0, 0]],
                                    susc_component='+-',
                                    write_eig='eig',
                                    filename='susc_+-_bse_',
                                    **kwargs):
        """Returns and writes real and imaginary part of the magnetic
        susceptibility.

        susc_componenet: str
            Component of the susceptibility tensor. '+-' and '-+'
            are supported.
        modes_Gc: list
            List of reciprocal lattice vectors in reduced on which the
            susceptibility is calculated. Default is the [0, 0, 0]
            component which gives the response over the Brillouin zone,
            but for optical magnons other components are needed.
        """
        self.susc_component = susc_component
        assert susc_component in ['+-', '-+']
        chi_Gw = []
        for mode_c in modes_Gc:
            assert len(mode_c) == 3
            assert all(isinstance(x, int) for x in mode_c)
            file_G = filename + ''.join(str(x) for x in mode_c) + '.csv'
            eig = write_eig + ''.join(str(x) for x in mode_c) + '.dat'
            vchi = self.vchi(*args, optical=False, mode_c=mode_c,
                             write_eig=eig, **kwargs)
            chi_Gw.append(vchi.magnetic_susceptibility(filename=file_G))
        return chi_Gw

    def vchi(self, w_w=None, eta=0.1, write_eig='eig.dat',
             optical=None, mode_c=None):
        vchi_w = self.get_vchi(w_w=w_w, eta=eta, optical=optical,
                               write_eig=write_eig, mode_c=mode_c)
        return VChi(self.gs.cd, self.context, w_w, vchi_w, optical=optical)

    def collect_A_SS(self, A_sS):
        comm = self.context.comm
        if comm.rank == 0:
            A_SS = np.zeros((self.nS, self.nS), dtype=complex)
            A_SS[:len(A_sS)] = A_sS
            Ntot = len(A_sS)
            for rank in range(1, comm.size):
                buf = np.empty((self.ns, self.nS), dtype=complex)
                comm.receive(buf, rank, tag=123)
                A_SS[Ntot:Ntot + self.ns] = buf
                Ntot += self.ns
        else:
            comm.send(A_sS, 0, tag=123)
        comm.barrier()
        if comm.rank == 0:
            return A_SS

    def parallelisation_kpoints(self, rank=None):
        comm = self.context.comm
        if rank is None:
            rank = comm.rank
        nK = self.kd.nbzkpts
        myKsize = -(-nK // comm.size)
        myKrange = range(rank * myKsize,
                         min((rank + 1) * myKsize, nK))
        myKsize = len(myKrange)
        return myKrange, myKsize

    def print_initialization(self, td, eshift, gw_kn):
        isl = ['----------------------------------------------------------',
               f'{self.mode} Hamiltonian',
               '----------------------------------------------------------',
               f'Started at:  {ctime()}', '',
               'Atoms                          : '
               f'{self.gs.atoms.get_chemical_formula(mode="hill")}',
               f'Ground state XC functional     : {self.gs.xcname}',
               f'Valence electrons              : {self.gs.nvalence}',
               f'Spinor calculations            : {self.add_soc}',
               f'Number of bands                : {self.gs.bd.nbands}',
               f'Number of spins                : {self.gs.nspins}',
               f'Number of k-points             : {self.kd.nbzkpts}',
               f'Number of irreducible k-points : {self.kd.nibzkpts}',
               f'Number of q-points             : {self.qd.nbzkpts}',
               f'Number of irreducible q-points : {self.qd.nibzkpts}', '']

        for q in self.qd.ibzk_kc:
            isl.append(f'    q: [{q[0]:1.4f} {q[1]:1.4f} {q[2]:1.4f}]')
        isl.append('')
        if gw_kn is not None:
            isl.append('User specified BSE bands')
        isl.extend([f'Response PW cutoff             : {self.ecut * Hartree} '
                    f'eV',
                    f'Screening bands included       : {self.nbands}'])
        isl.extend([f'Valence bands                  : {self.val_m}',
                    f'Conduction bands               : {self.con_m}'])
        if eshift is not None:
            isl.append(f'Scissors operator              : {eshift * Hartree}'
                       f'eV')
        isl.extend([
            f'Tamm-Dancoff approximation     : {td}',
            f'Number of pair orbitals        : {self.nS}',
            '',
            f'Truncation of Coulomb kernel   : {self.coulomb.truncation}'])
        integrate_gamma = self.integrate_gamma.type
        if self.integrate_gamma.reduced:
            integrate_gamma += '2D'
        isl.append(
            f'Coulomb integration scheme     : {integrate_gamma}')
        worldsize = self.context.comm.size
        isl.extend([
            '',
            '----------------------------------------------------------',
            '----------------------------------------------------------',
            '',
            f'Parallelization - Total number of CPUs   : {worldsize}',
            '  Screened potential',
            f'    K-point/band decomposition           : {worldsize}',
            '  Hamiltonian',
            f'    Pair orbital decomposition           : {worldsize}'])
        self.context.print('\n'.join(isl))


class BSE(BSEBackend):
    def __init__(self, calc=None, timer=None, txt='-', comm=world, **kwargs):
        """Creates the BSE object

        calc: str or calculator object
            The string should refer to the .gpw file contaning KS orbitals
        ecut: float
            Plane wave cutoff energy (eV)
        nbands: int
            Number of bands used for the screened interaction
        valence_bands: list or integer
            Valence bands used in the BSE Hamiltonian
        conduction_bands: list or integer
            Conduction bands used in the BSE Hamiltonian
        deps_max: float or None
            Maximum absolute value of transition energy for pair
            to be included in the BSE Hamiltonian
        add_soc: bool
            If True the calculation will included non-selfconsitent SOC.
            All band indices m refers to spinors, while n indices refer to
            states without SOC.
        scale: float
            Scaling of the SOC. A value of scale=1.0 yields a proper SOC
            calculation (id add_soc=True), whereas soc_scale=0 is equivalent
            to having add_soc=False.
        soc_tol: float
            Tolerance for how many non-SOC states are included when the SOC
            states are constructed (if add_soc=True). The SOC pair densities
            are constructed as linear combinations of pair densities without
            SOC. We include all states that contribute by more than soc_tol
            to the corresponding soc eigenstates.
        eshift: float
            Scissors operator opening the gap (eV)
        q_c: list of three floats
            Wavevector in reduced units on which the response is calculated
        direction: int
            if q_c = [0, 0, 0] this gives the direction in cartesian
            coordinates - 0=x, 1=y, 2=z
        gw_kn: list / array
            List or array defining the gw quasiparticle energies in eV
            used in the BSE Hamiltonian. Should match k-points and
            valence + conduction bands
        truncation: str or None
            Coulomb truncation scheme. Can be None or 2D.
        integrate_gamma: dict
        q0_correction: bool
            Whether to use analytical correction at q=0 in the
            calculation of W, applicable for 2D systems.
            Will raise an error if used without truncation='2D'
        txt: str
            txt output
        mode: str
            Theory level used. can be RPA TDHF or BSE. Only BSE is screened.
        """
        gs, context = get_gs_and_context(
            calc, txt, world=comm, timer=timer)

        super().__init__(gs=gs, context=context, **kwargs)


def write_bse_eigenvalues(filename, mode, w_w, C_w):
    with open(filename, 'w') as fd:
        print('# %s eigenvalues (in eV) and weights' % mode, file=fd)
        print('# Number   eig   weight', file=fd)
        for iw, (w, C) in enumerate(zip(w_w, C_w)):
            print('%8d %12.6f %12.16f' % (iw, w.real, C.real),
                  file=fd)


def read_bse_eigenvalues(filename):
    _, w_w, C_w = np.loadtxt(filename, unpack=True)
    return w_w, C_w


def write_spectrum(filename, w_w, A_w):
    with open(filename, 'w') as fd:
        for w, A in zip(w_w, A_w):
            print(f'{w:.9f}, {A:.9f}', file=fd)


def read_spectrum(filename):
    w_w, A_w = np.loadtxt(filename, delimiter=',',
                          unpack=True)
    return w_w, A_w


@dataclass
class VChi:
    cd: CellDescriptor
    context: ResponseContext
    w_w: np.ndarray
    vchi_w: np.ndarray
    optical: bool

    def epsilon(self):
        assert self.optical
        return -self.vchi_w + 1.0

    def eels(self):
        assert not self.optical
        return -self.vchi_w.imag

    def alpha(self):
        assert self.optical
        L = self.cd.nonperiodic_hypervolume
        return -L * self.vchi_w / (4 * np.pi)

    def susceptibility(self):
        assert not self.optical
        return self.vchi_w

    def dielectric_function(self, filename='df_bse.csv'):
        """Returns and writes real and imaginary part of the dielectric
        function.

        w_w: list of frequencies (eV)
            Dielectric function is calculated at these frequencies
        eta: float
            Lorentzian broadening of the spectrum (eV)
        filename: str
            data file on which frequencies, real and imaginary part of
            dielectric function is written
        write_eig: str
            File on which the BSE eigenvalues are written
        """

        return self._hackywrite(self.epsilon(), filename)

    # XXX The default filename clashes with that of dielectric function!
    def eels_spectrum(self, filename='df_bse.csv'):
        """Returns and writes real and imaginary part of the dielectric
        function.

        w_w: list of frequencies (eV)
            Dielectric function is calculated at these frequencies
        eta: float
            Lorentzian broadening of the spectrum (eV)
        filename: str
            data file on which frequencies, real and imaginary part of
            dielectric function is written
        write_eig: str
            File on which the BSE eigenvalues are written
        """
        return self._hackywrite(self.eels(), filename)

    def polarizability(self, filename='pol_bse.csv'):
        r"""Calculate the polarizability alpha.
        In 3D the imaginary part of the polarizability is related to the
        dielectric function by Im(eps_M) = 4 pi * Im(alpha). In systems
        with reduced dimensionality the converged value of alpha is
        independent of the cell volume. This is not the case for eps_M,
        which is ill defined. A truncated Coulomb kernel will always give
        eps_M = 1.0, whereas the polarizability maintains its structure.
        pbs should be a list of booleans giving the periodic directions.

        By default, generate a file 'pol_bse.csv'. The three colomns are:
        frequency (eV), Real(alpha), Imag(alpha). The dimension of alpha
        is \AA to the power of non-periodic directions.
        """
        return self._hackywrite(self.alpha(), filename)

    def magnetic_susceptibility(self, filename='susc_+-_0_bse.csv'):
        return self._hackywrite(self.susceptibility(), filename)[1]

    def _hackywrite(self, array, filename):
        comm = self.context.comm
        if comm.rank == 0 and filename is not None:
            if array.dtype == complex:
                write_response_function(filename, self.w_w, array.real,
                                        array.imag)
            else:
                assert array.dtype == float
                write_spectrum(filename, self.w_w, array)

        comm.barrier()

        self.context.print('Calculation completed at:', ctime(), flush=False)
        self.context.print('')

        return self.w_w, array


class BSEPlus:

    def create_chi0_full_calculator(self):
        chi0calc_full = Chi0Calculator(self.gs, self.context,
                                       wd=self.wd,
                                       nbands=self.rpa_nbands,
                                       intraband=False,
                                       hilbert=False,
                                       eta=self.eta,
                                       ecut=self.ecut,
                                       eshift=self.eshift)

        return chi0calc_full

    def create_bse_calculator(self):
        bse = BSE(self.bse_gpw,
                  ecut=self.ecut,
                  valence_bands=self.bse_valence_bands,
                  conduction_bands=self.bse_conduction_bands,
                  nbands=self.bse_nbands,
                  eshift=self.eshift,
                  mode='BSE',
                  truncation=self.truncation,
                  q_c=self.q_c,
                  direction=self.bse_direction,
                  add_soc=self.bse_add_soc,
                  txt='bse_calculation.txt',
                  comm=self.comm)

        return bse

    def create_chi0_limited_calculator(self):
        # XXX changed from world to self.context.comm.  --askhl
        self.gs, self.context = get_gs_and_context(
            self.rpa_gpw, txt=None, world=self.comm, timer=None)
        self.wd = get_frequency_descriptor(
            self.w_w, gs=self.gs, nbands=self.rpa_nbands)
        chi0calc_limited = Chi0Calculator(self.gs, self.context,
                                          wd=self.wd,
                                          nbands=slice(
                                              self.n1_chi0, self.m2_chi0),
                                          intraband=False,
                                          hilbert=False,
                                          eta=self.eta,
                                          ecut=self.ecut,
                                          eshift=self.eshift)
        return chi0calc_limited

    def get_chi_RPA(self, chi0calc, q_c, coulomb_kernel, xc_kernel,
                    CellDescriptor, direction):
        self.chi0_data = chi0calc.calculate(q_c)
        dyson_eqs = Chi0DysonEquations(self.chi0_data, coulomb_kernel,
                                       xc_kernel, CellDescriptor)
        self.v_G = coulomb_kernel.V(self.chi0_data.qpd)
        chi0_wGG = dyson_eqs.get_chi0_wGG(direction)
        chi0_WGG = dyson_eqs.wblocks.all_gather(chi0_wGG)
        del chi0calc, dyson_eqs, chi0_wGG
        return chi0_WGG

    def __init__(self,
                 bse_gpw,
                 bse_valence_bands,
                 bse_conduction_bands,
                 bse_nbands,
                 rpa_gpw,
                 rpa_nbands,
                 w_w,
                 eshift=0.0,
                 bse_add_soc=False,
                 eta=0.1,
                 q_c=(0.0, 0.0, 0.0),
                 direction=0,
                 truncation=None,
                 ecut=10,
                 comm=world):

        """ BSE+ calculation of chi. BSE+ offers a way to improve
        the convergence of the BSE by including transitions outside
        the active BSE electron-hole subspace at the RPA level in
        the irreducible polarizability. It saves the chi matrix
        calculated with the BSE+, BSE and RPA as npy-files.

        Parameters
        ----------
        bse_gpw: Path or str
            Name of the calculator that the BSE calculation should be
            made from (typically a fixed density calculator with less
            kpts)
        bse_valence_bands: range or list of integers
            Number of valence bands to be included in the bse calculation
        bse_conduction_bands: range or list of integers
            Number of conduction bands to be included in the bse calculation
        bse_nbands: integer
            Number of bands used for the screened interaction
        rpa_nbands: integer
            Number of bands to be included in the RPA calculation.
        w_w: list of floats
            Dielectric function is calculated at these frequencies (eV)
        eshift: float
            Scissors operator opening the gap (eV)
        bse_add_soc: bool
            If True the calculation will included non-self-consitent SOC in the
            underlying BSE calculation. SOC is not implemented in the RPA code.
        eta: float
            Lorentzian broadening of the spectrum (eV)
        q_c: list of three floats
            Wavevector in reduced units on which the response is calculated
        direction: int
            If q_c = [0, 0, 0] this gives the direction in cartesian
            coordinates - 0=x, 1=y, 2=z
        truncation: str or None
            Coulomb truncation scheme. Can be None or 2D.
        ecut: float
            Plane wave cutoff energy (eV)
         """

        self.bse_gpw = bse_gpw
        self.bse_valence_bands = bse_valence_bands
        self.bse_conduction_bands = bse_conduction_bands
        self.bse_nbands = bse_nbands
        self.rpa_gpw = rpa_gpw
        self.rpa_nbands = rpa_nbands
        self.w_w = w_w
        self.eshift = eshift
        self.bse_add_soc = bse_add_soc
        self.eta = eta
        self.q_c = list(q_c)
        self.bse_direction = direction
        self.rpa_direction = ('x', 'y', 'z')[direction]
        self.truncation = truncation
        self.ecut = ecut
        self.comm = comm

        self.n1_BSE = self.bse_valence_bands[0]
        self.m2_BSE = self.bse_conduction_bands[-1]
        self.m2_chi0_full = rpa_nbands - 1

        if bse_add_soc:
            self.n1_chi0 = int(self.n1_BSE / 2)
            self.m2_chi0 = int((self.m2_BSE + 1) / 2)
        else:
            self.n1_chi0 = self.n1_BSE
            self.m2_chi0 = self.m2_BSE + 1

        assert truncation in [None, '2D']

        assert self.m2_chi0 < self.m2_chi0_full, \
            'Large chi0 calculation should contain more ' \
            'bands than the BSE calculation'

    def calculate_chi_wGG(self, optical=True, xc_kernel=None,
                          bsep_name='chi_BSEPlus',
                          save_chi_BSE=False, save_chi_RPA=False):

        # irreducibale bse chi
        bse = self.create_bse_calculator()
        chi_irr_BSE_WGG = bse.get_chi_wGG(
            eta=self.eta,
            optical=optical,
            irreducible=True,
            w_w=self.w_w)
        del bse

        # chi0 calculation with the same bands as in the bse
        chi0calc_limited = self.create_chi0_limited_calculator()
        coulomb_kernel = CoulombKernel.from_gs(self.gs,
                                               truncation=self.truncation)
        chi0_limited_WGG = self.get_chi_RPA(chi0calc_limited, self.q_c,
                                            coulomb_kernel, xc_kernel,
                                            self.gs.cd, self.rpa_direction)

        # chi0 fully converged
        chi0calc_full = self.create_chi0_full_calculator()
        chi0_full_WGG = self.get_chi_RPA(chi0calc_full, self.q_c,
                                         coulomb_kernel, xc_kernel,
                                         self.gs.cd, self.rpa_direction)

        if self.truncation == '2D':
            pbc_c = self.gs.pbc
            assert sum(pbc_c) == 2
            coulomb_kernel_bare = CoulombKernel.from_gs(
                self.gs, truncation=None)
            v_G_bare = coulomb_kernel_bare.V(self.chi0_data.qpd, q_v=None)
            self.v_G = self.v_G / v_G_bare
            if optical:
                v_G_bare[0] = 0.0
            chi0_limited_WGG = chi0_limited_WGG * \
                v_G_bare[np.newaxis, np.newaxis, :]
            chi0_full_WGG = chi0_full_WGG * v_G_bare[np.newaxis, np.newaxis, :]
            chi_irr_BSE_WGG = chi_irr_BSE_WGG * \
                v_G_bare[np.newaxis, np.newaxis, :]
            cell_cv = self.gs.gd.cell_cv
            V = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
            V *= Bohr
        elif self.truncation is None and optical:
            self.v_G[0] = 0.0

        del self.chi0_data

        nR = len(self.w_w)
        # XXX Should this be another communicator?
        self.blocks = Blocks1D(self.comm, nR)

        chi_irr_BSE_wGG = chi_irr_BSE_WGG[self.blocks.myslice]
        chi0_full_wGG = chi0_full_WGG[self.blocks.myslice]
        chi0_limited_wGG = chi0_limited_WGG[self.blocks.myslice]

        chi_irr_BSEPlus_wGG = \
            chi_irr_BSE_wGG - chi0_limited_wGG + chi0_full_wGG
        eye = np.eye(chi_irr_BSEPlus_wGG.shape[1])

        chi_BSEPlus_wGG = \
            np.linalg.solve(eye - chi_irr_BSEPlus_wGG @ np.diag(self.v_G),
                            chi_irr_BSEPlus_wGG)

        if self.truncation == '2D':
            chi_BSEPlus_wGG *= V / (4 * np.pi)

        chi_BSEPlus_WGG = self.blocks.gather(chi_BSEPlus_wGG, 0)

        if self.comm.rank == 0:
            np.save(bsep_name + '.npy', chi_BSEPlus_WGG)
            del chi_BSEPlus_WGG
        del chi_BSEPlus_wGG, chi0_limited_wGG

        if save_chi_BSE:
            chi_BSE_wGG = \
                np.linalg.solve(eye - chi_irr_BSE_wGG @ np.diag(self.v_G),
                                chi_irr_BSE_wGG)

            if self.truncation == '2D':
                chi_BSE_wGG *= V / (4 * np.pi)

            chi_BSE_WGG = self.blocks.gather(chi_BSE_wGG, 0)

            if self.comm.rank == 0:
                np.save('chi_BSE.npy' if save_chi_BSE is True else
                        save_chi_BSE, chi_BSE_WGG)
                del chi_BSE_WGG
            del chi_BSE_wGG

        if save_chi_RPA:
            chi_full_wGG = \
                np.linalg.solve(eye - chi0_full_wGG @ np.diag(self.v_G),
                                chi0_full_wGG)

            if self.truncation == '2D':
                chi_full_wGG *= V / (4 * np.pi)

            chi_full_WGG = self.blocks.gather(chi_full_wGG, 0)

            if self.comm.rank == 0:
                np.save('chi_RPA.npy' if save_chi_RPA is True else
                        save_chi_RPA, chi_full_WGG)
                del chi_full_WGG
            del chi_full_wGG
