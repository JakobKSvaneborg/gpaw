import numpy as np

from gpaw.new.backwards_compatibility import FakeHamiltonian, FakeWFS
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.directmin.scf_helper import check_eigensolver_state, do_if_converged
from gpaw.directmin.etdm_fdpw import FDPWETDM


class ETDM(Eigensolver):
    def __init__(self,
                 *,
                 log,
                 preconditioner_factory,
                 excited_state: bool = False,
                 converge_unocc: bool = False):
        self.preconditioner = None
        self.preconditioner_factory = preconditioner_factory
        self.search_dir = LBFGS()
        # ibzwfs, density, potential,
        #                pot_calc, occ_calc, hamiltonian, mixer,
        #                log):
        # self.pot_calc = pot_calc
        #self.occ_calc = occ_calc
        self.log = log
        self.iters = 0
        self.alpha = 1.0  # step length
        self.energy_i = [None, None]  # energy at last two iterations
        self.dedalpha_i = [None, None]  # energy gradient w.r.t. alpha
        self.grad_knX = None

    def iterate(self,
                ibzwfs,
                density,
                potential,
                hamiltonian: Hamiltonian) -> float:
        if self.iters == 0:
            for wfs in ibzwfs:
                wfs.orthonormalize()
            grad = self.get_gradients_2(ham, wfs)
            self.project_gradient(wfs, grad)
            self.error = self.error_eigv(wfs, grad)
            self.eg_count += 1
            return energy, grad

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


class LBFGS:
    def __init__(self):
        from gpaw.directmin.sd_etdm import LBFGS as _LBFGS
        self._algo = _LBFGS()

    def update(self, ...):
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
