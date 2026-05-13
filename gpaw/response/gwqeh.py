from __future__ import division, print_function

import sys
from math import pi
import pickle

import numpy as np

from ase.units import Hartree, Bohr
from ase.dft.kpoints import monkhorst_pack

import gpaw.mpi as mpi
from gpaw.old.kpt_descriptor import KPointDescriptor
from gpaw.response.hilbert import HilbertTransform
from gpaw.response.g0w0 import select_kpts
from gpaw.response.groundstate import ResponseGroundStateAdapter
from gpaw.response.context import ResponseContext
from gpaw.response.pair import (KPointPairFactory, ActualPairDensityCalculator,
                                phase_shifted_fft_indices)
from gpaw.response.qpd import SingleQPWDescriptor


def frequency_grid(domega0, omega2, omegamax):
    beta = (2**0.5 - 1) * domega0 / omega2
    wmax = int(omegamax / (domega0 + beta * omegamax)) + 2
    w = np.arange(wmax)
    omega_w = w * domega0 / (1 - beta * w)
    return omega_w


# Hard-coded toggle: when True, GWQEHCorrection.calculate_W_QEH routes to
# the legacy single-basis qeh.old_qeh.Heterostructure path instead of the
# modern multi-basis qeh.QEH wrapper. Used internally to reproduce
# previously published GWQEH numbers against the same screening engine
# they were originally generated with. Not exposed as a public knob.
_USE_LEGACY_QEH = False


class GWQEHCorrection:
_PLACEHOLDER_REST_OF_FILE_