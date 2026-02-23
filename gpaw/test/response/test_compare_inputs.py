import numpy as np

from gpaw.response.g0w0 import compare_inputs


def test_compare_inputs(mpi):
    A = {'A': [[1.0 + 0.9e-14, 10], np.array([1.0 + 0.9e-14, 10])],
         'C': [1, 2, 3]}
    B = {'A': [[1.0, 10], np.array([1.0, 10])], 'C': [1, 2, 3]}
    assert compare_inputs(A, B)
