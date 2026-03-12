import numpy as np
from gpaw import GPAW_NO_C_EXTENSION
from gpaw.core import PWDesc, UGDesc
from gpaw.lfc import BasisFunctions
from gpaw.mpi import serial_comm
from gpaw.new.basis_functions import build_lfc_system
from gpaw.new.basis_functions_purepython import \
    BasisFunctionCollectionPurePython
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
                 band_comm=serial_comm,
                 xp=np,
                 gpu_add_and_integrate=True,
                 new_basis=False):
    kd = ibz._old_kd(nspins, kpt_comm)
    if GPAW_NO_C_EXTENSION:
        return SimpleBasis(grid, setups, relpos_ac, xp)
    basis_dtype = complex if \
        np.issubdtype(dtype, np.complexfloating) else float

    basis: BasisFunctions | BasisFunctionCollectionPurePython
    if not new_basis:
        basis = BasisFunctions(
            grid._gd,
            [setup.basis_functions_J for setup in setups],
            kd,
            dtype=basis_dtype,
            cut=True,
            xp=xp,
            gpu_add_and_integrate=gpu_add_and_integrate)
        basis.set_positions(relpos_ac)

    else:
        system = build_lfc_system(setups, grid, relpos_ac)
        # Only the purepython version is implemented for now
        basis = BasisFunctionCollectionPurePython(
            system,
            use_gpu=(xp is not np))

        assert grid.comm.size == 1
        basis.my_atom_indices = np.arange(len(relpos_ac))

    myM = (basis.Mmax + band_comm.size - 1) // band_comm.size
    basis.set_matrix_distribution(
        min(band_comm.rank * myM, basis.Mmax),
        min((band_comm.rank + 1) * myM, basis.Mmax))
    basis.grid = grid
    return basis


class SimpleBasis:
    """Quick hack for use without C-extensions."""
    def __init__(self,
                 grid: UGDesc,
                 setups,
                 relpos_ac,
                 xp):
        self.grid = grid
        self.pw = PWDesc(cell=grid.cell,
                         ecut=min(12.5, grid.ekin_max()))
        self.phit_aIG = self.pw.atom_centered_functions(
            [setup.basis_functions_J for setup in setups],
            relpos_ac)
        self.xp = xp

    def add_to_density(self,
                       nt_sR: np.ndarray,
                       f_asi):
        if self.xp is not np:
            _nt_sR = self.xp.asnumpy(nt_sR)
        else:
            _nt_sR = nt_sR
        nI = sum(f_si.shape[1] for f_si in f_asi.values())
        c_aiI = self.phit_aIG.empty(nI)
        c_aiI.data[:] = np.eye(nI)
        phit_IG = self.pw.zeros(nI)
        self.phit_aIG.add_to(phit_IG, c_aiI)
        I = 0
        for f_si in f_asi.values():
            for f_s in f_si.T:
                phit_R = phit_IG[I].ifft(grid=self.grid)
                _nt_sR += f_s[:, np.newaxis, np.newaxis, np.newaxis] * (
                    phit_R.data**2)
                I += 1
        if self.xp is not np:
            nt_sR[:] = self.xp.asarray(_nt_sR)
