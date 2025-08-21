import numpy as np
from ase import Atoms
from ase.build import fcc111
from gpaw import GPAW, PW
from gpaw.occupations import FermiDirac
from gpaw.mpi import rank
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.mft import HeisenbergExchangeCalculator
from gpaw.response.site_data import (AtomicSites, get_site_radii_range,
                                     calculate_site_magnetization)

# Setting up the AFM NiO
a0 = 4.17
a = fcc111('Ni', size=(1, 1, 3), a=a0)
pos_av = a.get_positions()
cell = a.get_cell()
a1 = pos_av[2] - pos_av[0]
a2 = a1 + cell[0]
a3 = a1 + cell[1]
bulk = Atoms('Ni2O2',
             scaled_positions=([0, 0, 0],
                               [0.5, 0.5, 0.5],
                               [0.25, 0.25, 0.25],
                               [0.75, 0.75, 0.75]),
             cell=(a1, a2, a3),
             pbc=True)
magmoms_a = [2, -2, 0, 0]
bulk.set_initial_magnetic_moments(magmoms_a)

# Calculations with 60 converged bands
Nb = 60
Nk = 18
calc = GPAW(mode=PW(1000),
            xc='LDA',
            occupations=FermiDirac(width=0.001),
            nbands=80,
            convergence={'density': 1.0e-6, 'bands': Nb},
            kpts={'size': (Nk, Nk, Nk), 'gamma': True},
            parallel={'domain': 1},
            txt='gs.txt',
            )
bulk.calc = calc
bulk.get_potential_energy()

# Setting up the LGZFG band path
qLG_qc = np.array([[1 / 2 - x / Nk, 0, 0]
                   for x in range(Nk // 2 + 1)])
qGZ_qc = np.array([[x / Nk, x / Nk, x / Nk]
                   for x in range(Nk // 2 + 1)])
qZF_qc = np.array([[1 / 2, 1 / 2, 1 / 2 - x / Nk]
                   for x in range(Nk // 2 + 1)])
qFG_qc = np.array([[1 / 2 - x / Nk, 1 / 2 - x / Nk, 0]
                   for x in range(Nk // 2 + 1)])
q_qc = np.vstack([qLG_qc, qGZ_qc[1:], qZF_qc[1:], qFG_qc[1:]])

# Convert relative q-points into distance along the bandpath
B_cv = 2.0 * np.pi * bulk.cell.reciprocal()
q_qv = q_qc @ B_cv
pathq_q = [0.]
for q in range(1, len(q_qc)):
    pathq_q.append(pathq_q[-1] + np.linalg.norm(q_qv[q] - q_qv[q - 1]))
pathq_q = np.array(pathq_q)
if rank == 0:
    np.save('q_qc.npy', q_qc)
    np.save('pathq_q.npy', pathq_q)

# Initialize response calculation
context = ResponseContext(txt='mft.txt')
gs = ResponseGroundStateAdapter(calc)
atoms = gs.atoms

# Initialize the spherical sites at maximal Ni radius
_, r_a = get_site_radii_range(gs)
r0 = np.min([r_a[0], r_a[1]])
sites = AtomicSites(indices=[0, 1], radii=[[r0], [r0]])
m_a = calculate_site_magnetization(gs, sites)[:, 0]

# Compute the isotropic exchange coupling along the chosen bandpath
jcalc = HeisenbergExchangeCalculator(gs,
                                     sites,
                                     context=context,
                                     nbands=Nb,
                                     nblocks=8)
J_qab = np.array([jcalc(q_c).array[..., 0] for q_c in q_qc])
context.write_timer()

# Save the bandpath, computed exchange constants and magnetic moments
if rank == 0:
    np.save('J_qab.npy', J_qab)
    np.save('m_a.npy', m_a)

# Redo everything with U = 5.5 eV
calc = GPAW(mode=PW(1000),
            xc='LDA',
            occupations=FermiDirac(width=0.001),
            nbands=80,
            setups={'Ni': ':d,5.5'},
            convergence={'density': 1.0e-6, 'bands': Nb},
            kpts={'size': (Nk, Nk, Nk), 'gamma': True},
            parallel={'domain': 1},
            txt='gs_U.txt',
            )
bulk.calc = calc
bulk.get_potential_energy()

context = ResponseContext(txt='mft_U.txt')
gs = ResponseGroundStateAdapter(calc)
m_a = calculate_site_magnetization(gs, sites)[:, 0]
jcalc = HeisenbergExchangeCalculator(gs,
                                     sites,
                                     context=context,
                                     nbands=Nb,
                                     nblocks=8)
J_qab = np.array([jcalc(q_c).array[..., 0] for q_c in q_qc])
context.write_timer()

if rank == 0:
    np.save('J_U_qab.npy', J_qab)
    np.save('m_U_a.npy', m_a)
