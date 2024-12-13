from functools import partial

import numpy as np
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomDistribution
from gpaw.new import zips
from gpaw.new.density import Density
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.potential import Potential
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunction
from gpaw.new.pwfd.lbfgs import LBFGS
from gpaw.setup import Setups


class ETDM(Eigensolver):
    def __init__(self,
                 *,
                 setups: Setups,
                 atomdist: AtomDistribution,
                 preconditioner_factory,
                 nspins: int,
                 excited_state: bool = False,
                 converge_unocc: bool = False,
                 xp=np):
        self.preconditioner = preconditioner_factory(10, xp=xp)
        self.search_dir = LBFGS()
        self.alpha = 1.0  # step length
        self.energy_i = (np.nan, np.nan)  # energy at last two iterations
        self.dedalpha_i = (np.nan, np.nan)  # energy gradient w.r.t. alpha
        self.grad_unX: list[XArray] = []
        self.dS_aii = setups.get_overlap_corrections(atomdist, xp)
        self.nocc_s = [-1] * nspins

    def iterate(self,
                ibzwfs: PWFDIBZWaveFunction,
                density: Density,
                potential: Potential,
                hamiltonian: Hamiltonian,
                pot_calc) -> float:
        dH = potential.dH
        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)

        if len(self.grad_unX) == 0:
            self.nocc_s = find_number_of_ocupied_bands(ibzwfs)

            for wfs in ibzwfs:
                wfs.orthonormalize()
                wfs.subspace_diagonalize(Ht, dH)

            energy, potential = update_density_and_potential(
                density, potential, pot_calc, ibzwfs, hamiltonian)
            Ht = partial(hamiltonian.apply,
                         potential.vt_sR,
                         potential.dedtaut_sR,
                         ibzwfs, density.D_asii)

            error = 0.0
            for wfs in ibzwfs:
                nocc = self.nocc_s[wfs.spin]
                psit_nX = wfs.psit_nX[:nocc]
                grad_nX = psit_nX.new()
                Ht(psit_nX, out=grad_nX, spin=0)
                apply_non_local_hamiltonian(grad_nX, wfs, potential)
                grad_nX.data *= wfs.myocc_n[:nocc, np.newaxis]
                project_gradient(grad_nX, wfs, self.dS_aii)
                error += grad_nX.norm2().sum()
                self.grad_unX.append(grad_nX)

        psit_unX = []
        for wfs in ibzwfs:
            nocc = self.nocc_s[wfs.spin]
            psit_nX = wfs.psit_nX[:nocc]
            psit_unX.append(psit_nX)

        pg_unX = []
        for psit_nX, grad_nX in zips(psit_unX, self.grad_unX):
            pg_nX = grad_nX.new()
            self.preconditioner(psit_nX, grad_nX, out=pg_nX)
            pg_nX.data *= -1.0 / (2 * (3 - len(self.nocc_s)))
            pg_unX.append(pg_nX)

        p_unX = self.search_dir.update(psit_unX, pg_unX)

        for wfs, p_nX in zips(ibzwfs, p_unX):
            project_gradient(p_nX, wfs)

        self.der_phi_2i[0] = sum(
            sum(p_X.integrate(grad_X)
                for p_X, grad_X in zip(p_nX, grad_nX)
                for p_nX, grad_nX in zips(p_unX, self.grad_unX)))

        for psit_nX, p_nX in zips(psit_unX, p_unX):
            psit_nX.data += alpha * p_nX.data

        for wfs in ibzwfs:
            wfs.orthonormalize()

        energy, potential = update_density_and_potential(
            density, potential, pot_calc, ibzwfs, hamiltonian)

            Ht = partial(hamiltonian.apply,
                         potential.vt_sR,
                         potential.dedtaut_sR,
                         ibzwfs, density.D_asii)

            error = 0.0
            for wfs in ibzwfs:
                nocc = self.nocc_s[wfs.spin]
                psit_nX = wfs.psit_nX[:nocc]
                grad_nX = psit_nX.new()
                Ht(psit_nX, out=grad_nX, spin=0)
                apply_non_local_hamiltonian(grad_nX, wfs, potential)
                grad_nX.data *= wfs.myocc_n[:nocc, np.newaxis]
                project_gradient(grad_nX, wfs, self.dS_aii)
                error += grad_nX.norm2().sum()
                self.grad_unX.append(grad_nX)
        phi, grad_k = self.get_energy_and_tangent_gradients(
            ham, wfs, dens, psit_knG=x_knG)

        der_phi = 0.0
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            for i, g in enumerate(grad_k[k]):
                    der_phi += self.dot(
                        wfs, g, search_dir[k][i], kpt,

        alpha, phi_alpha, der_phi_alpha, grad_knG = step_length_update(
            psit_unX, p_unX, wfs, ham, dens, converge_unocc,
                phi_0=phi_2i[0], der_phi_0=der_phi_2i[0],
                phi_old=phi_2i[1], der_phi_old=der_phi_2i[1],
                alpha_max=3.0, alpha_old=alpha, kpdescr=wfs.kd)
        self.alpha = alpha
        self.grad_knG = grad_knG

        # and 'shift' phi, der_phi for the next iteration
        phi_2i[1], der_phi_2i[1] = phi_2i[0], der_phi_2i[0]
        phi_2i[0], der_phi_2i[0] = phi_alpha, der_phi_alpha,

        self.iters += 1
        if not converge_unocc:
            self.globaliters += 1
        wfs.timer.stop('Direct Minimisation step')

        """
        e_entropy = 0.0
        kin_en_using_band = False
        e_sic = 0.0
        ham.get_energy(
            e_entropy, wfs, kin_en_using_band=kin_en_using_band, e_sic=e_sic)
        """
        return error

    def postprocess(self, ibzwfs, density, potential, hamiltonian):
        """wfs, ham, dens = self.whd(ibzwfs, density, potential, hamiltonian)
        do_if_converged(
            'etdm-fdpw', wfs, ham, dens, self.log)
        """
        ...


def apply_non_local_hamiltonian(Htpsit_nX,
                                wfs,
                                potential: Potential) -> None:
    nocc = len(Htpsit_nX)
    c_ani = {}
    dH_asii = potential.dH_asii
    for a, P_ni in wfs.P_ani.items():
        dH_ii = dH_asii[a][wfs.spin]
        c_ani[a] = P_ni[:nocc] @ dH_ii
    wfs.pt_aiX.add_to(Htpsit_nX, c_ani)


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


def update_density_and_potential(density,
                                 potential,
                                 pot_calc,
                                 ibzwfs,
                                 hamiltonian) -> tuple[float, Potential]:
    density.update(ibzwfs, ked=pot_calc.xc.type == 'MGGA')
    potential, _ = pot_calc.calculate(density, ibzwfs, potential.vHt_x)
    energy = (sum(e
                  for name, e in potential.energies.items()
                  if name not in ['stress', 'kinetic']) +
              sum(e
                  for name, e in ibzwfs.energies.items()
                  if name != 'band'))
    energy += ibzwfs.calculate_kinetic_energy(hamiltonian, density)
    return energy, potential


def find_number_of_ocupied_bands(ibzwfs: PWFDIBZWaveFunction) -> list[int]:
    nocc_s = [-1] * ibzwfs.nspins
    for wfs in ibzwfs:
        nocc = (wfs.occ_n > 0.5).sum()
        n = nocc_s[wfs.spin]
        if n != -1:
            assert nocc == n
        else:
            nocc_s[wfs.spin] = nocc
    return nocc_s


def step_length_update(x, p, evaluate_phi_and_der_phi, wfs,
                       max_step=0.2):
    slength = 0.0
    for k in p:
        for val in p[k]:
            slength += np.real(
                wfs.integrate(val, val, global_integral=False))
    slength = wfs.world.sum_scalar(slength)**0.5
    # slength = kd.comm.max_scalar(slength)

    a_star = max_step / slength if slength > max_step else 1.0

    phi_star, der_phi_star, g_star = evaluate_phi_and_der_phi(
        x, p, a_star, wfs)

    return a_star, phi_star, der_phi_star, g_star
