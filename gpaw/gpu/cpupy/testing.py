import numpy as np

from gpaw.gpu import cpupy as cp


def assert_allclose(actual,
                    desired,
                    rtol=1e-07,
                    atol=0,
                    err_msg='',
                    verbose=True) -> None:
    """"""
    np.testing.assert_allclose(cp.asnumpy(actual),
                               cp.asnumpy(desired),
                               rtol=rtol,
                               atol=atol,
                               err_msg=err_msg,
                               verbose=verbose)
