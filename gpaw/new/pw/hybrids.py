from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import pi, nan

import numpy as np
from gpaw.core import PWArray, PWDesc, UGArray, UGDesc
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.new import zips as zip
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.typing import Array1D
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.blas import mmm


