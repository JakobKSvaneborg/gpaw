import numpy as np
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomDistribution
from gpaw.new import zips
from gpaw.new.density import Density
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.potential import Potential
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunction
from gpaw.setup import Setups
from functools import partial


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
        self.preconditioner = None
        self.preconditioner = preconditioner_factory(10, xp=xp)
        self.search_dir = LBFGS()
        self.iters = 0
        self.alpha = 1.0  # step length
        self.energy_i = [None, None]  # energy at last two iterations
        self.dedalpha_i = [None, None]  # energy gradient w.r.t. alpha
        self.grad_unX: list = []
        self.dS_aii = setups.get_overlap_corrections(atomdist, xp)
        self.nocc_s = [-1] * nspins

    def iterate(self,
                ibzwfs: PWFDIBZWaveFunction,
                density: Density,
                potential: Potential,
                hamiltonian: Hamiltonian) -> float:
        dH = potential.dH
        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)
        if self.iters == 0:
            self.nocc_s = find_number_of_ocupied_bands(ibzwfs)
            error = 0.0
            for wfs in ibzwfs:
                wfs.orthonormalize()
                Htpsit_nX = wfs.psit_nX.new()
                wfs.subspace_diagonalize(Ht, dH,
                                         Htpsit_nX=Htpsit_nX)
                nocc = self.nocc_s[wfs.spin]
                print(nocc)
                grad_nX = Htpsit_nX[:nocc]
                apply_non_local_hamiltonian(grad_nX, wfs, potential)
                grad_nX.data *= wfs.myocc_n[:nocc, np.newaxis]
                project_gradient(grad_nX, self.dS_aii, wfs)
                self.grad_unX.append(grad_nX)
            error += grad_nX.norm2().sum()
            print(error)

        self.search_dir.update([wfs.psit_nX.copy() for wfs in ibzwfs],
                               self.grad_unX,
                               self.preconditioner)

        e_entropy = 0.0
        kin_en_using_band = False
        e_sic = 0.0
        ham.get_energy(
            e_entropy, wfs, kin_en_using_band=kin_en_using_band, e_sic=e_sic)
        return self.eigensolver.error

    def postprocess(self, ibzwfs, density, potential, hamiltonian):
        wfs, ham, dens = self.whd(ibzwfs, density, potential, hamiltonian)
        do_if_converged(
            'etdm-fdpw', wfs, ham, dens, self.log)


def apply_non_local_hamiltonian(Htpsit_nX,
                                wfs,
                                potential: Potential) -> None:
    nocc = len(Htpsit_nX)
    c_ani = {}
    dH_asii = potential.dH_asii
    for a, P_ni in wfs.P_ani.items():
        dH_ii = dH_asii[a][wfs.spin]
        c_ani[a] = P_ni[:nocc] @ dH_ii
        print(a, c_ani[a])
    wfs.pt_aiX.add_to(Htpsit_nX, c_ani)


def project_gradient(grad_nX: XArray,
                     dS_aii,
                     wfs) -> XArray:
    nocc = len(grad_nX)
    psit_nX = wfs.psit_nX[:nocc]

    M_nn = grad_nX.integrate(psit_nX)
    M_nn += M_nn.T.conj()
    M_nn *= 0.5
    print(M_nn)
    grad_nX.data -= M_nn @ psit_nX.data
    c_ani = {}
    for a, P_ni in wfs.P_ani.items():
        c_ani[a] = M_nn @ P_ni[:nocc] @ dS_aii[a]
    wfs.pt_aiX.add_to(grad_nX, c_ani)


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


class LBFGS:
    def __init__(self):
        from gpaw.directmin.sd_etdm import LBFGS as _LBFGS
        self._algo = _LBFGS()

    def update(self):
        self._algo.update_data()


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
