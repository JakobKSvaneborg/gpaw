"""This module defines an ELF function."""

import sys

import numpy as np
from gpaw.core import UGArray
from gpaw.fd_operators import Gradient
from gpaw.new.ase_interface import GPAW, ASECalculator
from gpaw.new.calculation import DFTCalculation


def _elf(nt_sg: np.ndarray,
         nt_grad2_sg: np.ndarray,
         taut_sg: np.ndarray,
         ncut: float,
         spinpol: bool) -> np.ndarray:
    """Pseudo electron localisation function (ELF).

    See:

      Becke and Edgecombe, J. Chem. Phys., vol 92 (1990) 5397

    More comprehensive definition in
    M. Kohout and A. Savin, Int. J. Quantum Chem., vol 60 (1996) 875-882

    Arguments:
    =============== =====================================================
    ``nt_sg``       Pseudo valence density.
    ``nt_grad2_sg`` Squared norm of the density gradient.
    ``tau_sg``      Kinetic energy density.
    ``ncut``        Minimum density cutoff parameter.
    ``spinpol``     Boolean indicator for spin polarization.
    =============== =====================================================
    """

    # Fermi constant
    cF = 3.0 / 10 * (3 * np.pi**2)**(2.0 / 3.0)

    if spinpol:
        # Kouhut eq. (9)
        D0 = 2**(2.0 / 3.0) * cF * (nt_sg[0]**(5.0 / 3.0) +
                                    nt_sg[1]**(5.0 / 3.0))

        taut = taut_sg.sum(axis=0)
        D = taut - (nt_grad2_sg[0] / nt_sg[0] + nt_grad2_sg[1] / nt_sg[1]) / 8
    else:
        # Kouhut eq. (7)
        D0 = cF * nt_sg[0]**(5.0 / 3.0)
        taut = taut_sg[0]
        D = taut - nt_grad2_sg[0] / nt_sg[0] / 8

    elf_g = 1.0 / (1.0 + (D / D0)**2)

    if ncut is not None:
        nt = nt_sg.sum(axis=0)
        elf_g[nt < ncut] = 0.0

    return elf_g


def elf_from_dft(dft: DFTCalculation | ASECalculator,
                 ncut: float = 1e-6) -> UGArray:
    """Calculate the electronic localization function.

    ``ncut``: density cutoff below which the ELF is zero.
    """
    density = dft.state.density
    density.update_ked(dft.state.ibzwfs)
    taut_sR = density.taut_sR
    nt_sR = density.nt_sR
    grad_v = [Gradient(nt_sR.desc._gd, v, n=2) for v in range(3)]
    gradnt2_sR = nt_sR.new(zeroed=True)
    for gradnt2_R, nt_R in zip(gradnt2_sR, nt_sR):
        for grad in grad_v:
            gradnt_R = grad(nt_R)
            gradnt2_R.data += gradnt_R.data**2
    elf_R = nt_sR.desc.empty()
    elf_R.data[:] = _elf(nt_sR.data, gradnt2_sR.data, taut_sR.data,
                         ncut, spinpol=len(nt_sR) == 2)
    return elf_R


if __name__ == '__main__':
    elf_from_dft(GPAW(sys.argv[1]).dft).isosurface()
