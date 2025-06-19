import numpy as np
from ase.units import _e
from ase.parallel import paropen
from gpaw import GPAW
from gpaw.mpi import serial_comm
from gpaw.berryphase import polarization_phase
from pathlib import Path

gpw_gs = Path("BaTiO3.gpw")
gpw_wfs = Path("BaTiO3+wfs.gpw")

# create gpw-file with wave functions for all k-points in the BZ
calc = GPAW(gpw_gs).fixed_density(symmetry="off")
calc.write(gpw_wfs, mode="all")

phases_c = polarization_phase(gpw_wfs, comm=serial_comm)
phi_c = phases_c["phase_c"]

cell_cv = calc.atoms.cell * 1e-10  # in m
vol = calc.atoms.get_volume() * 1e-30  # in m^3

# phase defined modulo 2 pi
pi2 = 2 * np.pi
phi_c -= (phi_c / pi2) * pi2

# polarization in C/m^2
px, py, pz = phi_c @ cell_cv / vol * _e
with paropen("polarization_BaTiO3.out", "w") as fd:
    fd.write(f"P: {pz} C/m^2\n")
