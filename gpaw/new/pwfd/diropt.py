from __future__ import annotations

from functools import partial

from gpaw.core.arrays import XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.new import trace, zips
from gpaw.new.density import Density
from gpaw.new.energies import DFTEnergies
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.potential import Potential
from gpaw.new.pwfd.eigensolver import PWFDEigensolver
from gpaw.new.etdm.searchdir import LBFGS
from gpaw.new.pwfd.multixarray import MultiXArray


class DirOptPWFD(PWFDEigensolver):
    def __init__(self,
                 *,
                 hamiltonian,
                 nbands,
                 domain_band_comm,
                 excited_state: bool = False,
                 converge_unocc: bool = False,
                 convergence: dict,
                 alpha: float = 0.2,
                 scalapack_params=(None, 1, 1, None)):

        # Lazy initialization of search_dir, done later in iterate()
        self.search_dir: LBFGS | None = None
        self.grad_unX: list[XArray] = []
        self.dS_aii: AtomArrays
        self.nocc_s: list[int] = []
        self.scalapack = scalapack_params
        self.alpha = alpha
        self.converge_unocc = converge_unocc
        super().__init__(
            hamiltonian=hamiltonian,
            convergence=convergence,
            domain_band_comm=domain_band_comm,
            nbands=nbands,
            scalapack_parameters=scalapack_params)

    def new(self, **params) -> DirOptPWFD:
        return DirOptPWFD(**params)

    @trace
    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density: Density,
                potential: Potential,
                hamiltonian: Hamiltonian,
                pot_calc,
                energies) -> tuple[float, float, DFTEnergies]:

        # no band parallization supported
        assert ibzwfs.band_comm.size == 1

        hamiltonian.update_wave_functions(ibzwfs)

        if len(self.nocc_s) == 0:
            # init: setup preconditioner
            self._initialize(ibzwfs)
            # xp: type of distributed wfs array
            # np -> numpy
            # cp -> cupy
            xp = ibzwfs.xp

            # self.nocc_s = find_number_of_occupied_bands(ibzwfs)
            self.nocc_s = find_highest_occupied_bands(ibzwfs)

            if self.converge_bands == 'occupied':
                self.nband_s = self.nocc_s.copy()
            elif self.converge_bands == 'all':
                self.nband_s = [ibzwfs.nbands for _ in self.nocc_s]
            else:
                assert isinstance(self.converge_bands, int)
                self.nband_s = [self.converge_bands for _ in self.nocc_s]

            self.dS_aii = pot_calc.setups.get_overlap_corrections(
                density.D_asii.layout.atomdist, xp)

        nspins = len(self.nocc_s)

        # H_KS = - 1/2 nabla^2 + veff(r) + dExc/dtau O_tau
        #                        vt_sR     dedtaut_sR (projection |tau><tau|)
        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)

        # build wfs
        psit_unX = build_wfs(ibzwfs, self.nocc_s)

        if len(self.grad_unX) == 0:
            # build first gradient vector
            orthogonalize(ibzwfs)
            update_eigenvalues(ibzwfs, Ht, potential,
                               nband_s=self.nocc_s,
                               eigenvalues_only=True)

            # update density and hamiltonian
            energies, potential = update_density_and_potential(
                density, potential, pot_calc, ibzwfs, hamiltonian)

            Ht = partial(hamiltonian.apply,
                         potential.vt_sR,
                         potential.dedtaut_sR,
                         ibzwfs, density.D_asii)

            # get gradient by applying hamiltonian
            self.grad_unX = apply_hamiltonian(ibzwfs, psit_unX, Ht, potential)

            # project gradient
            for grad_nX, wfs in zips(self.grad_unX, ibzwfs):
                project_wfs(grad_nX, wfs, dS_aii=self.dS_aii)

            # precondition gradient
            self.grad_unX = precondition(psit_unX, self.grad_unX,
                                         self.preconditioner, nspins)

        if self.search_dir is None:
            self.search_dir = LBFGS()

        weights = [wfs.weight for wfs in ibzwfs]

        p_unX = self.search_dir.update(
            MultiXArray(psit_unX, ibzwfs.kpt_comm, weights),
            MultiXArray(self.grad_unX, ibzwfs.kpt_comm, weights)).a_unX

        # for wfs, p_nX in zips(ibzwfs, p_unX):
        #     # projecting search direction on tangent space at psi
        #     # is slightly different from project gradient
        #     # as it doesn't apply overlap matrix because of S^{-1}
        #     project_wfs(p_nX, wfs)

        # # total projected search_direction length
        # slength = ibzwfs.kpt_comm.sum_scalar(
        #     sum(p_nX.norm2().sum() for p_nX in p_unX))**0.5
        # max_step = 0.2
        # alpha = max_step / slength if slength > max_step else 1.0

        # update wavefunctions coefficents
        for psit_nX, p_nX in zips(psit_unX, p_unX):
            psit_nX.data += self.alpha * p_nX.data

        # orthongonalize wavefunctions
        orthogonalize(ibzwfs)

        # update density
        energies, potential = update_density_and_potential(
            density, potential, pot_calc, ibzwfs, hamiltonian)

        # update hamiltonian
        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)

        # get gradient by applying hamiltonian
        self.grad_unX = apply_hamiltonian(ibzwfs, psit_unX, Ht, potential)

        # project gradient
        for grad_nX, wfs in zips(self.grad_unX, ibzwfs):
            project_wfs(grad_nX, wfs, dS_aii=self.dS_aii)

        # precondition gradient
        self.grad_unX = precondition(psit_unX, self.grad_unX,
                                     self.preconditioner, nspins)

        error = 0.0
        # calculate residual
        for grad_nX, wfs in zip(self.grad_unX, ibzwfs):
            nbands = len(grad_nX)
            # weights according to kpt, spin and occupation f_n
            weight_n = (wfs.weight * wfs.spin_degeneracy *
                        wfs.myocc_n[:nbands])
            # update gradient with weights
            shape = (-1,) + (1,) * (grad_nX.data.ndim - 1)
            grad_nX.data *= weight_n.reshape(shape)
            # sum weigthed residual
            error += grad_nX.norm2().sum()
        error = ibzwfs.kpt_comm.sum_scalar(error)

        return 0.0, error, energies

    def postprocess(self, ibzwfs, density, potential, hamiltonian,
                    maxiter, cc, log):

        nspins = len(self.nocc_s)

        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)

        orthogonalize(ibzwfs)
        update_eigenvalues(ibzwfs, Ht, potential,
                           nband_s=self.nocc_s,
                           eigenvalues_only=self.converge_unocc)

        # reset search direction
        self.search_dir.reset()
        self.grad_unX = []

        if not self.converge_unocc:
            return

        import numpy as np
        import time

        # build first gradient
        # build wfs with bands to converge
        psit_unX = build_wfs(ibzwfs, self.nband_s)
        self.grad_unX = apply_hamiltonian(ibzwfs, psit_unX, Ht, potential)

        # project gradient
        for grad_nX, wfs in zips(self.grad_unX, ibzwfs):
            project_wfs(grad_nX, wfs, dS_aii=self.dS_aii)

        # precondition gradient
        self.grad_unX = precondition(psit_unX, self.grad_unX,
                                     self.preconditioner, nspins)

        niter = 0
        log('Converging unoccupied states')
        while niter < maxiter:

            weights_u = [wfs.weight for wfs in ibzwfs]

            p_unX = self.search_dir.update(
                MultiXArray(psit_unX, ibzwfs.kpt_comm, weights_u),
                MultiXArray(self.grad_unX, ibzwfs.kpt_comm, weights_u)).a_unX

            # update wavefunctions coefficents
            for psit_nX, p_nX in zips(psit_unX, p_unX):
                psit_nX.data += self.alpha * p_nX.data

            orthogonalize(ibzwfs)
            # update gradient
            self.grad_unX = apply_hamiltonian(ibzwfs, psit_unX, Ht, potential)

            # project gradient
            for grad_nX, wfs in zips(self.grad_unX, ibzwfs):
                project_wfs(grad_nX, wfs, dS_aii=self.dS_aii)

            # precondition gradient
            self.grad_unX = precondition(psit_unX, self.grad_unX,
                                         self.preconditioner, nspins)

            error = 0.0
            # calculate residual
            for grad_nX, wfs in zip(self.grad_unX, ibzwfs):
                # weights according to kpt, spin
                # sum weigthed residual
                weights = wfs.weight * wfs.spin_degeneracy
                error += grad_nX.norm2().sum() * weights
            error = ibzwfs.kpt_comm.sum_scalar(error)

            # iterations and time.
            now = time.localtime()
            line = ('iter:{:4d} {:02d}:{:02d}:{:02d} '
                    .format(niter, *now[3:6]))
            # eigenstates
            line += 14 * ' ' + '{:+6.2f}'.format(np.log10(error))
            log(line)

            if abs(error) < cc['eigenstates'].tol:
                break

            niter += 1

        orthogonalize(ibzwfs)
        update_eigenvalues(ibzwfs, Ht, potential,
                           nband_s=self.nband_s)

        # reset search direction
        self.search_dir.reset()
        self.grad_unX = []


def build_wfs(ibzwfs, nband_s):
    psit_unX = []
    for wfs in ibzwfs:
        nband = nband_s[wfs.spin]
        bslice = slice(0, nband, 1)
        psit_nX = wfs.psit_nX[bslice]
        psit_unX.append(psit_nX)
    return psit_unX


def orthogonalize(ibzwfs):
    # check whether already orthogonal states are changed
    for wfs in ibzwfs:
        wfs._P_ani = None
        wfs.orthonormalized = False
        wfs.orthonormalize()


def update_eigenvalues(ibzwfs, Ht, potential,
                       nband_s, eigenvalues_only=False):
    for wfs in ibzwfs:
        nband = nband_s[wfs.spin]
        tmp_nX = wfs.psit_nX.new()
        wfs.subspace_diagonalize(Ht, potential.deltaH, tmp_nX,
                                 nocc=nband,
                                 eigenvalues_only=eigenvalues_only)


def apply_hamiltonian(ibzwfs, psit_unX, Ht, potential):
    grad_unX = []
    for psit_nX, wfs in zips(psit_unX, ibzwfs):
        grad_nX = psit_nX.new()
        Ht(psit_nX, out=grad_nX, spin=wfs.spin)
        apply_non_local_hamiltonian(grad_nX, wfs, potential)
        grad_unX.append(grad_nX)

    return grad_unX


@trace
def project_wfs(grad_nX: XArray, wfs,
                dS_aii=None):

    # gradient grad_nX
    # | g_nX > = H_KS | psit_nX >

    # project gradient
    # | pg_nX > = | g_nX > - < psi_mX | g_nX > S | psi_mX >
    #           = | g_nX > - M_mn S | psi_mX>
    # with M_mn = < psi_mX | g_nX >
    # such that < psi_mX | pg_nX> = 0
    # because < psi_mX | S | psi_mX > = 1
    nbands = len(grad_nX)
    bslice = slice(0, nbands, 1)

    psit_mX = wfs.psit_nX[bslice]
    mbands = psit_mX.data.shape[0]

    M_nm = grad_nX.integrate(psit_mX)
    # condition to satisfy: Psi^+ G + G^+ Psi = 0
    M_nm += M_nm.T.conj()
    M_nm *= 0.5

    # Reshape is needed here for FD-mode:
    grad_nX.data -= (M_nm @ psit_mX.data.reshape((mbands, -1))).reshape(
        grad_nX.data.shape)

    # PAW contribution
    if dS_aii is not None:
        c_ani = {}
        for a, P_mi in wfs.P_ani.items():
            c_ani[a] = M_nm @ P_mi[bslice] @ -dS_aii[a]
        wfs.pt_aiX.add_to(grad_nX, c_ani)


def precondition(psit_unX, grad_unX, preconditioner, nspins):
    pg_unX = []
    for psit_nX, grad_nX in zips(psit_unX, grad_unX):
        pg_nX = grad_nX.new()
        preconditioner(psit_nX, grad_nX, out=pg_nX)
        pg_nX.data *= -1.0 / (2 * (3 - nspins))
        pg_unX.append(pg_nX)
    return pg_unX


@trace
def apply_non_local_hamiltonian(Htpsit_nX,
                                wfs,
                                potential: Potential,
                                bands: slice | None = None) -> None:
    bands = bands or slice(len(Htpsit_nX))
    c_ani = {}
    dH_asii = potential.dH_asii
    for a, P_ni in wfs.P_ani.items():
        dH_ii = dH_asii[a][wfs.spin]
        c_ani[a] = P_ni[bands] @ dH_ii
    wfs.pt_aiX.add_to(Htpsit_nX, c_ani)


@trace
def update_density_and_potential(density,
                                 potential,
                                 pot_calc,
                                 ibzwfs,
                                 hamiltonian) -> tuple[float, Potential]:
    density.update(ibzwfs, ked=pot_calc.xc.type == 'MGGA')
    potential, energies, _ = pot_calc.calculate(density,
                                                ibzwfs,
                                                potential.vHt_x)
    energies.set(kinetic=ibzwfs.calculate_kinetic_energy(hamiltonian, density),
                 band=0.0)
    return energies, potential


def find_number_of_occupied_bands(ibzwfs: IBZWaveFunctions) -> list[int]:
    nocc_s = [-1] * ibzwfs.nspins
    for wfs in ibzwfs:
        nocc = (wfs.occ_n > 0.4).sum()
        n = nocc_s[wfs.spin]
        if n != -1:
            assert nocc == n
        else:
            nocc_s[wfs.spin] = nocc
    return nocc_s


def find_highest_occupied_bands(ibzwfs: IBZWaveFunctions) -> list[int]:
    nocc_s = [-1] * ibzwfs.nspins
    for wfs in ibzwfs:
        nocc = max(oo for oo, occ in enumerate(wfs.occ_n)
                   if occ > 0.4) + 1
        n = nocc_s[wfs.spin]
        if n != -1:
            assert nocc == n
        else:
            nocc_s[wfs.spin] = nocc
    return nocc_s
