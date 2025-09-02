import numpy as np
from gpaw.mpi import world

from .etdm import ETDM
from .edmiston_ruedenberg.obj_function import EdmistonRuedenbergUpdateRef as ER
from .skewherm_matrix import random_a


def er_localize(ibzwfs, state2opt="all"):
    for states in state2opt.split("-"):
        for loct in ["paw", "pseudo"]:
            objfunc = ER(ibzwfs, loct, states)
            a_init = []
            for a in objfunc.a_vec_u:
                skmat = random_a((a.ndim, a.ndim), ibzwfs.dtype) * 0.01
                skmat -= skmat.T.conj()
                a_init.append(skmat[a.ind_up])
            a_init = np.asarray(a_init)
            world.broadcast(a_init, 0)
            gtol = 1.0e-6
            niter = 500
            if loct == "paw":
                gtol = 1.0e-12
                niter = 2000
            etdm = ETDM(objfunc, a_init, niter, gtol, True)
            etdm.optimize()
