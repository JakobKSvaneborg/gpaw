import numpy as np

from gpaw.blacs import BlacsDescriptor, BlacsGrid, Redistributor
from gpaw.mpi import world

npw = 10351
npw = 5352
S = 8
nprow, npcol, b = 2, 4, 64
nprow, npcol, b = 2, 4, 16
r0 = world.rank // S * S
comm = world.new_communicator(range(r0, r0 + S))
bg = BlacsGrid(comm, S, 1)
bg2 = BlacsGrid(comm, nprow, npcol)
mynpw = -(-npw // S)
print(mynpw, S * mynpw - npw)
md = BlacsDescriptor(bg, npw, npw, mynpw, npw)
md2 = BlacsDescriptor(bg2, npw, npw, b, b)
H_GG = md.zeros(dtype=complex)
S_GG = md.zeros(dtype=complex)
G1, G2 = next(md.my_blocks(S_GG))[:2]
assert G1 == comm.rank * mynpw
H_GG.ravel()[G1::npw + 1] = world.rank + 1
S_GG.ravel()[G1::npw + 1] = 1.0
r = Redistributor(comm, md, md2)
H_GG = r.redistribute(H_GG)
S_GG = r.redistribute(S_GG)
psit_nG = md2.empty(dtype=complex)
eps_n = np.empty(npw)
eps_n[:] = 0.0  # np.nan
# md2.general_diagonalize_dc(H_GG, S_GG, psit_nG, eps_n)
md2.diagonalize_dc(H_GG, psit_nG, eps_n)
print(world.rank, comm.rank, eps_n.min(), eps_n.max())
assert eps_n.min() == r0 + 1
assert eps_n.max() == r0 + 8
