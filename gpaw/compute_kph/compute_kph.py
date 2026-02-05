from gpaw import GPAW
import numpy as np
from gpaw.response.bse import BSE
from gpaw.mpi import world
import pickle


g_unique_q = np.load('g_matrix_unique_q.npy', mmap_mode='r')

if world.rank == 0:
    print('loaded g', flush=True)

with open("C_phases.pkl", "rb") as f:
    C = pickle.load(f)

if world.rank == 0:
    print('loaded C', flush=True)

q_indices = np.load('q_indices.npy')

if world.rank == 0:
    print('loaded q', flush=True)

w_l = np.load('w_phonon_unique_q.npy')

if world.rank == 0:
    print('Starting Kph calculation', flush=True)

bse = BSE('scf_PW.gpw',
          ecut=50,
          valence_bands=6,
          conduction_bands=6,
          nbands=100,
          mode='BSE',
          txt='bse_Si.txt')


if world.rank == 0:
    print('made bse', flush=True)

kph = bse.construct_Kph(T = 0.000086*300, #0.026,
                       eta=0.1, g_unique_q=g_unique_q,
                      C_knm=C, w_phonon_unique_q=w_l,
                      q_indices=q_indices, A_foldername=True, eigidx=range(0,5))

if world.rank == 0:
    print('saving Kph')
    np.save('kph.npy', kph)
