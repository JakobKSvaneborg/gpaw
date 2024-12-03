from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import BasisFunctions
from gpaw.mpi import serial_comm
from gpaw.new.brillouin import IBZ


def create_basis(ibz: IBZ,
                 nspins,
                 pbc_c,
                 grid,
                 setups,
                 dtype,
                 relpos_ac,
                 comm=serial_comm,
                 kpt_comm=serial_comm,
                 band_comm=serial_comm):
    kd = KPointDescriptor(ibz.bz.kpt_Kc, nspins)

    kd.ibzk_kc = ibz.kpt_kc
    kd.weight_k = ibz.weight_k
    kd.sym_k = ibz.s_K
    kd.time_reversal_k = ibz.time_reversal_K
    kd.bz2ibz_k = ibz.bz2ibz_K
    kd.ibz2bz_k = ibz.ibz2bz_k
    kd.bz2bz_ks = ibz.bz2bz_Ks
    kd.nibzkpts = len(ibz)
    kd.symmetry = ibz.symmetries._old_symmetry
    kd.set_communicator(kpt_comm)

    basis = BasisFunctions(grid._gd,
                           [setup.basis_functions_J for setup in setups],
                           kd,
                           dtype=dtype,
                           cut=True)
    basis.set_positions(relpos_ac)
    myM = (basis.Mmax + band_comm.size - 1) // band_comm.size
    basis.set_matrix_distribution(
        min(band_comm.rank * myM, basis.Mmax),
        min((band_comm.rank + 1) * myM, basis.Mmax))
    return basis
