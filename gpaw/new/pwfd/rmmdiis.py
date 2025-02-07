from __future__ import annotations

from pprint import pformat

import numpy as np

# from gpaw.new import zips as zip
from gpaw.new.pwfd.eigensolver import PWFDEigensolver  # , calculate_residuals
from gpaw.core import PWDesc


class RMMDIIS(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 preconditioner_factory,
                 blocksize=None,
                 converge_bands='occupied',
                 scalapack_parameters=None):
        if blocksize is None:
            if isinstance(wf_grid, PWDesc):
                S = wf_grid.comm.size
                # Use a multiple of S for maximum efficiency
                blocksize = int(np.ceil(10 / S)) * S
            else:
                blocksize = 10
        super().__init__(preconditioner_factory, converge_bands, blocksize)

    def __str__(self):
        return pformat(dict(name='RMMDIIS',
                            converge_bands=self.converge_bands))

    def iterate1(self, wfs, Ht, dH, dS_aii, weight_n):
        """Do one step ...

        See the old implementation in:

            gpaw.eigensolvers.rmmdiis.RMMDIIS.iterate_one_k_point()

        Also take a look at the new Davidson eigensolver in this module:

            gpaw.new.pwfd.davidson

        Also take a look at ths document:

            https://gpaw.readthedocs.io/documentation/rmm-diis.html
        """
        raise NotImplementedError
