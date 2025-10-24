import numpy as np
from gpaw.mpi import world

from gpaw.new.etdm.etdm import ETDM
from gpaw.new.etdm.edmiston_ruedenberg.obj_function import (
    EdmistonRuedenbergUpdateRef as ER,
)
from gpaw.new.etdm.skewherm_matrix import random_a


def er_localize(ibzwfs,
                states="all",
                loct="pseudo-paw",
                gtol=1e-6,
                niter=333,
                seed=None
    ):
    objfunc = ER(ibzwfs, loct, states)
    a_init = []
    for a in objfunc.a_vec_u:
        skmat = random_a((a.ndim, a.ndim),
                          ibzwfs.dtype, seed=seed) * 0.01
        skmat -= skmat.T.conj()
        a_init.append(skmat[a.ind_up])
    a_init = np.asarray(a_init)
    world.broadcast(a_init, 0)

    etdm = ETDM(objfunc, a_init, niter, gtol, True)
    etdm.optimize()

    return etdm
