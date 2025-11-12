import numpy as np
from gpaw.core import UGArray
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.external import ConstantElectricField, ExternalPotential
from gpaw.fd_operators import Gradient, Laplace
from gpaw.new import zips
from gpaw.new.fd.pot_calc import FDPotentialCalculator
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.hamiltonian import Hamiltonian


class FDHamiltonian(Hamiltonian):
    def __init__(self, grid, *, kin_stencil=3, xp=np):
        self.grid = grid
        self._gd = grid._gd
        self.kin = Laplace(self._gd, -0.5, kin_stencil, grid.dtype, xp=xp)

        # For MGGA:
        self.grad_v = []

    def apply_local_potential(self,
                              vt_R: UGArray,
                              psit_nR: XArray,
                              out: XArray,
                              ) -> None:
        assert isinstance(psit_nR, UGArray)
        assert isinstance(out, UGArray)
        self.kin(psit_nR, out)
        for p, o in zips(psit_nR.data, out.data):
            o += p * vt_R.data

    def apply_mgga(self,
                   dedtaut_R: UGArray,
                   psit_nR: XArray,
                   vt_nR: XArray) -> None:
        if len(self.grad_v) == 0:
            grid = psit_nR.desc
            self.grad_v = [
                Gradient(grid._gd, v, n=3, dtype=grid.dtype)
                for v in range(3)]

        tmp_R = psit_nR.desc.empty()
        for psit_R, out_R in zips(psit_nR, vt_nR):
            for grad in self.grad_v:
                grad(psit_R, tmp_R)
                tmp_R.data *= dedtaut_R.data
                grad(tmp_R, tmp_R)
                tmp_R.data *= 0.5
                out_R.data -= tmp_R.data

    def create_preconditioner(self, blocksize, xp=np):
        from types import SimpleNamespace

        from gpaw.preconditioner import Preconditioner as PC
        pc = PC(self._gd, self.kin, self.grid.dtype, blocksize, xp=xp)

        def apply(psit, residuals, out, ekin_n=None):
            kpt = SimpleNamespace(phase_cd=psit.desc.phase_factor_cd)
            pc(residuals.data, kpt, out=out.data)

        return apply

    def calculate_kinetic_energy(self, wfs, skip_sum=False):
        e_kin = 0.0
        for f, psit_R in zip(wfs.myocc_n, wfs.psit_nX):
            if f > 1.0e-10:
                e_kin += f * psit_R.integrate(self.kin(psit_R), skip_sum).real
        if not skip_sum:
            e_kin = psit_R.desc.comm.sum_scalar(e_kin)
            e_kin = wfs.band_comm.sum_scalar(e_kin)
        return e_kin * wfs.spin_degeneracy


class FDKickHamiltonian(FDHamiltonian):

    def __init__(self,
                 grid,
                 ext: ExternalPotential,
                 ibzwfs: PWFDIBZWaveFunctions,
                 pot_calc: FDPotentialCalculator,
                 layout: AtomArraysLayout,
                 **kwargs):
        """ Factory class for creating a Hamiltonian-like object
        representing a potential kick """
        self.vext_R = grid.empty()
        if not isinstance(ext, ConstantElectricField):
            raise NotImplementedError

        r_Rv = grid.xyz()
        # This is a shifted grid, compared to ext.calculate_potential
        self.vext_R.data[:] = np.einsum('xyzv,v->xyz', r_Rv, ext.field_v)
        wfs = ibzwfs._wfs_u[0]
        positions_av = wfs.relpos_ac @ grid.cell_cv
        potential_a = positions_av @ ext.field_v

        # calculate coefficient
        # ---------------------
        #
        # coeffs_ni =
        #   P_nj * c0 * 1_ij
        #   + P_nj * cx * x_ij
        #
        # where (see spherical_harmonics.py)
        #
        #   1_ij = sqrt(4pi) Delta_0ij
        #   y_ij = sqrt(4pi/3) Delta_1ij
        #   z_ij = sqrt(4pi/3) Delta_2ij
        #   x_ij = sqrt(4pi/3) Delta_3ij
        # ...

        #   1_ij = sqrt(4pi) Delta_0ij
        #   y_ij = sqrt(4pi/3) Delta_1ij
        #   z_ij = sqrt(4pi/3) Delta_2ij
        #   x_ij = sqrt(4pi/3) Delta_3ij

        # coefficients
        # coefs_ni = sum_j ( <phi_i| f(x,y,z) | phi_j>
        #                    - <phit_i| f(x,y,z) | phit_j> ) P_nj

        assert wfs.nspins == 1
        self.dH_asii = layout.empty(wfs.nspins)
        for a, c0 in enumerate(potential_a):
            Delta_iiL = wfs.setups[a].Delta_iiL
            coef_ii = np.sqrt(4 * np.pi) * c0 * Delta_iiL[..., 0]
            coef_ii += np.sqrt(4 * np.pi / 3) \
                * ext.field_v[0] * Delta_iiL[..., 3]  # x
            coef_ii += np.sqrt(4 * np.pi / 3) \
                * ext.field_v[1] * Delta_iiL[..., 1]  # y
            coef_ii += np.sqrt(4 * np.pi / 3) \
                * ext.field_v[2] * Delta_iiL[..., 2]  # z
            self.dH_asii[a][0] = coef_ii

    def dH(self, P_ani, out_ani, spin):
        assert spin == 0
        assert len(P_ani.dims) == 1, 'only collinear wave functions'

        P_ani.block_diag_multiply(self.dH_asii, out_ani, spin)

    def apply_local_potential(self,
                              vt_R: UGArray,
                              psit_nR: XArray,
                              out: XArray,
                              ) -> None:
        assert isinstance(psit_nR, UGArray)
        assert isinstance(out, UGArray)
        # The supplied potential is ignored.
        # We are kicking with the potential stored by the init
        for p, o in zips(psit_nR.data, out.data):
            o += p * self.vext_R.data
