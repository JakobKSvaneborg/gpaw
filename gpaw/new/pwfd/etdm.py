import numpy as np

from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.core.arrays import DistributedArrays as XArray


class ETDM(Eigensolver):
    def __init__(self,
                 *,
                 preconditioner_factory,
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
        self.grad_knX = None
        self.dS_aii = setups.get_overlap_corrections(atomdist, xp)

    def iterate(self,
                ibzwfs,
                density,
                potential,
                hamiltonian: Hamiltonian) -> float:
        if self.iters == 0:
            error = 0.0
            for wfs in ibzwfs:
                wfs.orthonormalize()
                grad_nX = gradient(ibzwfs,
                                   density,
                                   potential,
                                   hamiltonian,
                                   wfs)
            error += calculate_error(grad_nX)

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


def gradient(hamiltonian,
             potential,
             ibzwfs,
             density,
             wfs):
    psit_nX = wfs.psit_nX
    grad_nX = psit_nX.new()

    hamiltonian.apply(
        potential.vt_sR,
        potential.dedtaut_sR,
        ibzwfs,  # used by hybrids
        density.D_asii,  # used by hybrids
        psit_nX,
        grad_nX,
        wfs.spin)

    c_ani = {}
    dH_asii = ham.potential.dH_asii
    for a, P_ni in wfs.P_ani.items():
        dH_ii = dH_asii[a][wfs.spin]
        c_ani[a] = P_ni @ dH_ii
    wfs.pt_aiX.add_to(grad_nX, c_ani)

    # scale with occupation numbers
    for f, grad_X in zips(wfs.myocc_n, grad_nX.data):
        grad_X *= f
    project_gradient(wfs, grad)


class LBFGS:
    def __init__(self):
        from gpaw.directmin.sd_etdm import LBFGS as _LBFGS
        self._algo = _LBFGS()

    def update(self):
        self._algo.update_data()


def get_slength(p, wfs, mode=None):
    if mode is None:
        mode = wfs.mode
    if mode == 'lcao':
        p_all_kpts = np.hstack([p[k] for k in p])
        return np.linalg.norm(p_all_kpts)
    else:
        ret = 0.0
        for k in p:
            for val in p[k]:
                ret += np.real(wfs.integrate(val, val, global_integral=False))
        ret = wfs.world.sum_scalar(ret)
        return np.sqrt(ret)


class MaxStep:

    def __init__(self, evaluate_phi_and_der_phi, max_step=0.2):
        """

        :param evaluate_phi_and_der_phi:
        """

        self.evaluate_phi_and_der_phi = evaluate_phi_and_der_phi
        self.max_step = max_step
        self.name = 'max-step'

    def todict(self):
        return {'name': self.name,
                'max_step': self.max_step}

    def step_length_update(self, x, p, wfs, *args, mode=None, **kwargs):

        kd = kwargs['kpdescr']

        slength = get_slength(p, wfs, mode)
        slength = kd.comm.max_scalar(slength)

        a_star = self.max_step / slength if slength > self.max_step else 1.0

        phi_star, der_phi_star, g_star = self.evaluate_phi_and_der_phi(
            x, p, a_star, wfs, *args)

        return a_star, phi_star, der_phi_star, g_star
