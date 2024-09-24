"""Calculate the site magnetization and exchange parameters of Co (hcp)."""

import numpy as np

from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.site_data import (AtomicSites, get_site_radii_range,
                                     maximize_site_magnetization)
from gpaw.response.mft import (calculate_exchange_parameters,
                               HeisenbergExchangeCalculator)

# ----- Ground state calculation ----- #

# Set up crystal structure
a = 2.5071
c = 4.0695
mm = 1.67
atoms = bulk('Co', 'hcp', a=a, c=c)
atoms.set_initial_magnetic_moments([mm, mm])
atoms.center()

# Perform ground state calculation
kpts = 24
nbands = 2 * 6  # natoms * (4s + 3d)
calc = GPAW(xc='LDA',
            mode=PW(800),
            kpts={'size': (kpts, kpts, kpts), 'gamma': True},
            nbands=nbands + 8,  # add some extra bands for smooth converence
            # We converge the ground state density tightly
            convergence={'density': 1e-8,
                         'eigenstates': 1e-14,
                         'bands': nbands},
            occupations=FermiDirac(0.001),
            parallel={'domain': 1},
            txt='Co_gs.txt')
atoms.calc = calc
atoms.get_potential_energy()
gs = ResponseGroundStateAdapter(calc)

# ----- Magnon energies as a function of rc ----- #

# Get the valid site radii range
rmin_a, rmax_a = get_site_radii_range(gs)
# which should be identical for the two Co atoms
assert abs(rmin_a[1] - rmin_a[0]) < 1e-6
assert abs(rmax_a[1] - rmax_a[0]) < 1e-6
rc_r = np.linspace(rmin_a[0], rmax_a[0], 51)  # partitionings
rc_ar = [rc_r] * 2

# Set up magnetic sites and calculate the exchange parameters at the
# high-symmetry points
context = ResponseContext('Co_hsp.txt')
sites = AtomicSites(indices=[0, 1], radii=rc_ar)
q_pc = np.array([[0., 0., 0],  # Γ
                 [0.5, 0., 0.],  # M
                 [1 / 3., 1 / 3., 0.],  # K
                 [0., 0., 0.5]])  # A
J_pabr = np.array(
    [calculate_exchange_parameters(
        gs, sites, q_c, context=context, nbands=nbands)
     for q_c in q_pc])

# ----- Magnon dispersion at ideal rc ----- #

# Find ideal cutoff radii
rc_a, magmom_a = maximize_site_magnetization(gs)
# which again should be the same for the two Co atoms due to symmetry
assert abs(rc_a[1] - rc_a[0]) < 1e-6
assert abs(magmom_a[1] - magmom_a[0]) < 1e-6
rc, magmom = rc_a[0], magmom_a[0]

# Define the commensurate q-vectors to calculate (in this case a
# G-M-K-G-A high-symmetry path)
qGM_qc = np.array([[x / kpts, 0., 0.]
                   for x in range(kpts // 2 + 1)])
qMK_qc = np.array([[1 / 2. - x / kpts, 2 * x / kpts, 0.]
                   for x in range(kpts // 6 + 1)])
qKG_qc = np.array([[x / kpts, x / kpts, 0.]
                   for x in reversed(range(kpts // 3 + 1))])
qGA_qc = np.array([[0., 0., x / kpts]
                   for x in range(kpts // 2 + 1)])
q_qc = np.vstack([qGM_qc, qMK_qc[1:], qKG_qc[1:], qGA_qc[1:]])

# Set up magnetic site and calculate J_ab(q)
context.new_txt_and_timer('Co_dispersion.txt')
sites = AtomicSites(indices=[0, 1], radii=[[rc], [rc]])
# When calculating the Heisenberg exchange for many values of q_c, it
# is beneficial to utilize underlying HeisenbergExchangeCalculator, to
# which calculate_exchange_parameters() is a single-use interface.
jcalc = HeisenbergExchangeCalculator(
    gs, sites, context=context, nbands=nbands)
J_qab = np.array([jcalc(q_c).array[..., 0]  # dimension: J_abp
                  for q_c in q_qc])
context.write_timer()

# ----- Save results ----- #

if context.comm.rank == 0:
    atoms.write('Co.json')
    np.save('Co_rc_r.npy', rc_r)
    np.save('Co_q_pc.npy', q_pc)
    np.save('Co_J_pabr.npy', J_pabr)
    np.save('Co_rc.npy', rc)
    np.save('Co_magmom.npy', magmom)
    np.save('Co_q_qc.npy', q_qc)
    np.save('Co_J_qab.npy', J_qab)
