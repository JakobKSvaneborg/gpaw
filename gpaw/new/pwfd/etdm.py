from __future__ import annotations
from functools import partial

import numpy as np
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.new import zips
from gpaw.new.density import Density
from gpaw.new.pwfd.eigensolver import PWFDEigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.potential import Potential
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.lbfgs import LBFGS
from gpaw.new.energies import DFTEnergies
from gpaw.new import trace


class ETDM(PWFDEigensolver):
    def __init__(self,
                 *,
                 hamiltonian,
                 excited_state: bool = False,
                 converge_unocc: bool = False):
        self.search_dir = LBFGS()
        self.grad_unX: list[XArray] = []
        self.converge_unocc = converge_unocc
        self.dS_aii: AtomArrays
        self.nocc_s: list[int] = []
        super().__init__(hamiltonian)

    def new(self, **params) -> ETDM:
        return ETDM(**params)

    @trace
    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density: Density,
                potential: Potential,
                hamiltonian: Hamiltonian,
                pot_calc,
                energies) -> tuple[float, float, DFTEnergies]:

        if len(self.nocc_s) == 0:
            # init: setup preconditioner
            self._initialize(ibzwfs)
            # xp: type of distributed wfs array
            # np -> numpy
            # cp -> cupy
            xp = ibzwfs.xp
            self.nocc_s = find_number_of_occupied_bands(ibzwfs)
            self.dS_aii = pot_calc.setups.get_overlap_corrections(
                density.D_asii.layout.atomdist, xp)

        dH = potential.dH
        # H_KS = - 1/2 nabla^2 + veff(r) + dExc/dtau O_tau
        #                        vt_sR     dedtaut_sR (projection |tau><tau|)
        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)

        if len(self.grad_unX) == 0:
            # build first gradient vector

            for wfs in ibzwfs:
                wfs._P_ani = None
                # why orthonormalize uninitialized tmp_nX?
                tmp_nX = wfs.psit_nX.new()
                wfs.orthonormalized = False
                wfs.orthonormalize(tmp_nX)
                wfs.subspace_diagonalize(Ht, dH, tmp_nX)

            energies, potential = update_density_and_potential(
                density, potential, pot_calc, ibzwfs, hamiltonian)
            Ht = partial(hamiltonian.apply,
                         potential.vt_sR,
                         potential.dedtaut_sR,
                         ibzwfs, density.D_asii)

            for wfs in ibzwfs:
                nocc = self.nocc_s[wfs.spin]
                psit_nX = wfs.psit_nX[:nocc]
                grad_nX = psit_nX.new()
                Ht(psit_nX, out=grad_nX, spin=wfs.spin)
                apply_non_local_hamiltonian(grad_nX, wfs, potential)
                # gradient grad_nX from residual
                # | R_nX > = H_KS | psit_nX >
                #          - Re(M_nn) | psit_nX >
                #          - sum_a M_nn @ P_ani @ dS_aii
                # with M_nn < psit_nX | H_KS | psit_nX >
                project_gradient(grad_nX, wfs, self.dS_aii)
                # weights according to kpt, spin and occupation f_n
                weight_n = (wfs.weight * wfs.spin_degeneracy *
                            wfs.myocc_n[:nocc])
                grad_nX.data *= weight_n[:, np.newaxis]
                self.grad_unX.append(grad_nX)

        psit_unX = []
        for wfs in ibzwfs:
            nocc = self.nocc_s[wfs.spin]
            psit_nX = wfs.psit_nX[:nocc]
            psit_unX.append(psit_nX)

        # precondition gradient
        pg_unX = []
        for psit_nX, grad_nX in zips(psit_unX, self.grad_unX):
            pg_nX = grad_nX.new()
            self.preconditioner(psit_nX, grad_nX, out=pg_nX)
            pg_nX.data *= -1.0 / (2 * (3 - len(self.nocc_s)))
            pg_unX.append(pg_nX)

        p_unX = self.search_dir.update(psit_unX, pg_unX)
        for wfs, p_nX in zips(ibzwfs, p_unX):
            # why no dS_aii term? what happens to paw correction?
            project_gradient(p_nX, wfs)

        # total projected search_direction length
        slength = sum(p_nX.norm2().sum() for p_nX in p_unX)**0.5
        max_step = 0.2
        alpha = max_step / slength if slength > max_step else 1.0

        # update wavefunctions coefficents
        for psit_nX, p_nX in zips(psit_unX, p_unX):
            psit_nX.data += alpha * p_nX.data

        # update wavefunctions
        for wfs in ibzwfs:
            wfs._P_ani = None
            wfs.orthonormalized = False
            wfs.orthonormalize()

        # update density
        energies, potential = update_density_and_potential(
            density, potential, pot_calc, ibzwfs, hamiltonian)

        # update hamiltonian
        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)

        error = 0.0
        # from updated hamiltonian and wfs calculate new (projected) residual
        for psit_nX, grad_nX, wfs in zips(psit_unX, self.grad_unX, ibzwfs):
            Ht(psit_nX, out=grad_nX, spin=wfs.spin)
            apply_non_local_hamiltonian(grad_nX, wfs, potential)
            project_gradient(grad_nX, wfs, self.dS_aii)
            weight_n = (wfs.weight * wfs.spin_degeneracy *
                        wfs.myocc_n[:nocc])
            # sum weigthed residual
            error += grad_nX.norm2() @ weight_n
            grad_nX.data *= weight_n[:, np.newaxis]

        return 0.0, error, energies

    def postprocess(self, ibzwfs, density, potential, hamiltonian):
        if not self.converge_unocc:
            return

        # dH = potential.dH
        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)

        grad_unX = []
        psit_unX = []
        for wfs in ibzwfs:
            nocc = self.nocc_s[wfs.spin]
            psit_nX = wfs.psit_nX[nocc:]
            psit_unX.append(psit_nX)
            grad_nX = psit_nX.new()
            Ht(psit_nX, out=grad_nX, spin=wfs.spin)
            apply_non_local_hamiltonian(grad_nX, wfs, potential,
                                        slice(nocc, None))
            project_gradient(grad_nX, wfs, self.dS_aii)
            weight = wfs.weight * wfs.spin_degeneracy
            grad_nX.data *= weight
            grad_unX.append(grad_nX)

        while 1:
            pg_unX = []
            for psit_nX, grad_nX in zips(psit_unX, grad_unX):
                pg_nX = grad_nX.new()
                self.preconditioner(psit_nX, grad_nX, out=pg_nX)
                pg_nX.data *= -1.0 / (2 * (3 - len(self.nocc_s)))
                pg_unX.append(pg_nX)

            p_unX = self.search_dir.update(psit_unX, pg_unX)
            for wfs, p_nX in zips(ibzwfs, p_unX):
                project_gradient(p_nX, wfs)

            slength = sum(p_nX.norm2().sum() for p_nX in p_unX)**0.5
            max_step = 0.2
            alpha = max_step / slength if slength > max_step else 1.0

            for psit_nX, p_nX in zips(psit_unX, p_unX):
                psit_nX.data += alpha * p_nX.data

            for wfs in ibzwfs:
                wfs._P_ani = None
                wfs.orthonormalized = False
                wfs.orthonormalize()

            error = 0.0
            for psit_nX, grad_nX, wfs in zips(psit_unX, grad_unX, ibzwfs):
                Ht(psit_nX, out=grad_nX, spin=wfs.spin)
                apply_non_local_hamiltonian(grad_nX, wfs, potential)
                project_gradient(grad_nX, wfs, self.dS_aii)
                weight = wfs.weight * wfs.spin_degeneracy
                error += grad_nX.norm2().sum() * weight
                grad_nX.data *= weight
            print(error)


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
def project_gradient(grad_nX: XArray,
                     wfs,
                     dS_aii=None):
    nocc = len(grad_nX)
    psit_nX = wfs.psit_nX[:nocc]

    M_nn = grad_nX.integrate(psit_nX)
    M_nn += M_nn.T.conj()
    M_nn *= 0.5
    grad_nX.data -= M_nn @ psit_nX.data
    if dS_aii:
        c_ani = {}
        for a, P_ni in wfs.P_ani.items():
            c_ani[a] = M_nn @ P_ni[:nocc] @ -dS_aii[a]
        wfs.pt_aiX.add_to(grad_nX, c_ani)

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
        nocc = (wfs.occ_n > 0.5).sum()
        n = nocc_s[wfs.spin]
        if n != -1:
            assert nocc == n
        else:
            nocc_s[wfs.spin] = nocc
    return nocc_s
