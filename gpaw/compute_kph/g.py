from gpaw import GPAW
from gpaw.elph import ElectronPhononMatrix
from gpaw.mpi import world
import numpy as np
import os

calc = GPAW("scf.gpw")
atoms = calc.atoms

#q = [[0., 0., 0.], [1. / 11., 1. / 11., 1. / 11.]]
q = np.load('q_list.npy')
chunk_size = 50
nq = len(q)
nq = q.shape[0]

q_chunks = [q[i:i + chunk_size] for i in range(0, nq, chunk_size)]

chunk_dir = "g_chunks"
if world.rank == 0 and not os.path.exists(chunk_dir):
    os.makedirs(chunk_dir)
world.barrier()


all_gs = []
for i, q_chunk in enumerate(q_chunks):
    chunk_file = f"{chunk_dir}/g_chunk_{i:04d}.npy"
    if world.rank == 0 and os.path.isfile(chunk_file):
        print(f"Skipping existing chunk {i}")
        continue

    if world.rank == 0:
        print(f"Processing q-chunk {i+1}/{len(q_chunks)} with {len(q_chunk)} q-points")

    elph = ElectronPhononMatrix(atoms, 'supercell', 'elph')

    g_sqklnn = elph.bloch_matrix(calc, k_qc=q_chunk,
                             savetofile=False, prefactor=False)

    if world.rank == 0:
        all_gs.append(g_sqklnn)

    if world.rank == 0:
        np.save(chunk_file, g_sqklnn)
        print(f"Saved {chunk_file}")

    world.barrier()
    
    del g_sqklnn
    del elph



if world.rank == 0:
    g_sqklnn_full = np.concatenate(all_gs, axis=1)
    print(f'{np.shape(g_sqklnn_full) = }')
    np.save('g_matrix_unique_q.npy', g_sqklnn_full)


    g_sqklnn = np.load('g_matrix_unique_q.npy')
 #   q_indices = np.load('q_indices.npy')

#    g_full = g_sqklnn[:, q_indices, ...]

 #   print(f'{np.shape(g_full) = }')

 #   np.save('g_matrix_all_q.npy', g_full)
#
