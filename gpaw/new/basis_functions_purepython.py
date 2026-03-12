from gpaw.new.basis_functions import (BasisFunctionCollectionBase,
                                      SplinePoolBase,
                                      BasisFunctionDesc)
from gpaw.gpu import cupy as cp, cupyx
from gpaw.sphere.spherical_harmonics import Y
from gpaw.typing import override

import numpy as np
from scipy.interpolate import CubicSpline


class SplinePoolPurePython(SplinePoolBase):
    """"""

    @override
    def add_spline(self, desc: BasisFunctionDesc) -> None:
        """"""
        spline = CubicSpline(
            np.linspace(0.0, desc.cutoff, len(desc.f_r), endpoint=True),
            desc.f_r,
            bc_type='clamped',  # first derivatives are zero at boundaries
            extrapolate=False)

        self.splines.append(spline)


class SplinePoolGPUPurePython(SplinePoolBase):

    @override
    def add_spline(self, desc: BasisFunctionDesc) -> None:
        """"""

        # Cupyx does not have a direct CubicSpline replacement, so we use
        # a PPoly object (generic piecewise polynomial).
        # TODO handle fake cupy?
        spline = CubicSpline(
            np.linspace(0.0, desc.cutoff, len(desc.f_r), endpoint=True),
            desc.f_r,
            bc_type='clamped',  # first derivatives are zero at boundaries
            extrapolate=False)

        spline = cupyx.scipy.interpolate.PPoly(
            cp.asarray(spline.c),
            cp.asarray(spline.x),
            extrapolate=False)

        self.splines.append(spline)


class BasisFunctionCollectionPurePython(BasisFunctionCollectionBase):
    """Pure Python implementation of BasisFunctionCollection.
    Not very optimized.
    """

    @override
    def _init_splines(self) -> None:
        """"""
        assert self.phi_datas
        on_gpu = self.xp is not np

        self.spline_pool = (
            SplinePoolGPUPurePython() if on_gpu else SplinePoolPurePython()
        )

        for phi_desc in self.phi_datas.values():
            self.spline_pool.add_spline(phi_desc)

        assert self.spline_pool.num_splines() == len(self.phi_datas.keys())

    def evaluate_spline(self, spline_idx: int, x: np.ndarray | cp.ndarray) \
            -> np.ndarray | cp.ndarray:
        """"""
        spline = self.get_spline(spline_idx)
        y = spline(x)
        # Purepython splines return NaN for x > xmax, should return 0.0
        return self.xp.nan_to_num(y, nan=0.0)

    @override
    def precalculate_on_grid(self) -> None:
        """
        """

        # TODO precalculation on GPU
        for block in self._block_map.values():
            num_m = len(block.M_m)

            phi_mg_shape = (num_m, *block.shape)
            block.evaluated_phi_mg = np.zeros(phi_mg_shape, dtype=np.float64)

            x_Gv = block.get_block_xyz()

            mu_local = 0
            for phi in block.phi_j:
                # Get distance from phi (atom) center for each XYZ point
                d_Gv = x_Gv - phi.position
                d_G = (d_Gv**2).sum(axis=3)**0.5

                f_r = self.evaluate_spline(phi.spline_index, d_G)

                l = phi.get_angular_momentum_number()
                # ensure the order is what we expect
                assert block.M_m[mu_local] == phi.first_mu

                for m in range(0, phi.get_num_mu()):
                    block.evaluated_phi_mg[mu_local + m] = (
                        f_r * Y(l**2 + m,
                                d_Gv[..., 0], d_Gv[..., 1], d_Gv[..., 2])
                    )
                mu_local += phi.get_num_mu()

            assert np.all(np.isfinite(block.evaluated_phi_mg))

    @override
    def has_precalculated_phi(self) -> bool:
        """"""
        return True

    @override
    def add_to_density(
        self,
        nt_sG: np.ndarray | cp.ndarray,
        f_asi: dict[int, np.ndarray] | dict[int, cp.ndarray]
    ) -> None:
        r"""Add linear combination of squared localized basis functions to
        density:

            nt_s(x) += \sum_a \sum_i f_{asi} |\phi_{ai}|^2

        where i runs over all basis functions for said atom.

        Parameters
        ----------
        nt_sG : np.ndarray | cp.ndarray
            The density array to which contributions are added. Must be domain
            aware: if (nx, ny, nz) is the shape of this MPI grid domain, nt_sG
            must have shape (num_spins, nx, ny, nz).
            Modified in-place.
        f_asi : dict[int, np.ndarray] | dict[int, cp.ndarray]
            Dictionary that maps atom indices to occupation coefficient
            arrays. Each array has shape (num_spins, n_a), if n_a is the
            number of basis functions for that atom.
        Raises
        ------
        AssertionError
            If input array/dict shapes are incorrect.
        """

        num_spins = nt_sG.shape[0]
        assert np.all(self.grid.mysize_c == nt_sG.shape[1:])

        if self.has_precalculated_phi():
            # Flatten f_asi dict to a single array, ordered by global mu
            f_sM = np.zeros((num_spins, self.Mmax))

            for atom_idx in range(self.num_atoms):
                atom_mu_range = self._mu_range_a[atom_idx]
                f_sM[:, atom_mu_range.start:atom_mu_range.stop] = (
                    f_asi[atom_idx]
                )

            for block in self._block_map.values():
                assert block.evaluated_phi_mg is not None

                # XYZ indices to the nt_sG array for this block (domain aware)
                start_offset_c = block.start_c - self.grid.start_c
                sx, sy, sz = start_offset_c
                ex, ey, ez = start_offset_c + block.get_block_shape()

                # Take XYZ slice to handle smaller boundary blocks
                local_start_c, local_end_c = block.get_local_start_end_c()
                phi2_mg = block.evaluated_phi_mg[
                    :,
                    local_start_c[0]:local_end_c[0],
                    local_start_c[1]:local_end_c[1],
                    local_start_c[2]:local_end_c[2]]**2

                for local_m, mu in enumerate(block.M_m):

                    for s in range(num_spins):
                        nt_sG[s, sx:ex, sy:ey, sz:ez] += (
                            f_sM[s, mu] * phi2_mg[local_m]
                        )

        else:
            raise NotImplementedError(
                "PurePython BasisFunctions without precalculation")

    @override
    def calculate_potential_matrix(
        self,
        vt_G: np.ndarray | cp.ndarray,
        out: np.ndarray | cp.ndarray | None = None
    ) -> np.ndarray | cp.ndarray:
        """"""
        xp = cp if isinstance(vt_G, cp.ndarray) else np

        assert np.all(vt_G.shape == self.grid.mysize_c)

        num_work_rows = (self._matrix_distribution_rules.mu_end
                         - self._matrix_distribution_rules.mu_start)
        if (num_work_rows <= 0):
            # nothing to do
            if out:
                return out
            else:
                return xp.empty((0))

        if out:
            if out.ndim != 2:
                raise ValueError("out array must be 2D and have enough rows")
            rows, cols = out.shape
            if rows < num_work_rows or cols != self.num_basis_functions():
                raise ValueError("Not enough rows or columns in out array")
            out[:] = 0

        M = self.num_basis_functions()
        res = out or xp.zeros((num_work_rows, M))

        if self.has_precalculated_phi():
            self._potential_matrix_with_precalculation(
                vt_G,
                res,
                self._matrix_distribution_rules.mu_start,
                self._matrix_distribution_rules.mu_end)
            return res
        else:
            raise NotImplementedError(
                "PurePython BasisFunctions without precalculation")

    def _potential_matrix_with_precalculation(
        self,
        vt_G: np.ndarray | cp.ndarray,
        out: np.ndarray | cp.ndarray,
        mu_start: int,
        mu_end: int
    ) -> None:
        """"""

        for block in self.get_relevant_blocks():
            assert block.evaluated_phi_mg is not None

            # Get slice of the potential in this block (vt_G is domain aware)
            start_c, end_c = block.get_domain_local_start_end_c()

            vt_g = vt_G[start_c[0]:end_c[0],
                        start_c[1]:end_c[1],
                        start_c[2]:end_c[2]]

            phi_mg = block.evaluated_phi_mg
            phi_mu_vt_g = phi_mg * vt_g
            # Integrate, ie. contract grid indices. Dense block-sized matrix
            V_mn = np.einsum('mxyz, nxyz -> mn', phi_mu_vt_g, phi_mg,
                             optimize=True)

            for m, mu in enumerate(block.M_m):
                for n, nu in enumerate(block.M_m):

                    if mu >= nu and mu in range(mu_start, mu_end):
                        # account for row-distributed output matrix
                        out[mu - mu_start, nu] += V_mn[m, n]
        #
        out *= self.grid.dv

    @override
    def construct_density(self,
                          rho_MM,
                          nt_G: np.ndarray | cp.ndarray,
                          q):
        """"""
        assert np.all(nt_G.shape == self.grid.mysize_c)

        num_work_rows = (self._matrix_distribution_rules.mu_end
                         - self._matrix_distribution_rules.mu_start)
        if num_work_rows <= 0:
            # nothing to do
            return

        if self.has_precalculated_phi():
            self._construct_density_with_precalculation(
                rho_MM,
                nt_G,
                self._matrix_distribution_rules.mu_start,
                self._matrix_distribution_rules.mu_end)
        else:
            raise NotImplementedError(
                "PurePython BasisFunctions without precalculation")

    def _construct_density_with_precalculation(
        self,
        rho_MM,
        nt_G: np.ndarray | cp.ndarray,
        mu_start: int,
        mu_end: int
    ) -> None:
        for block in self.get_relevant_blocks():
            phi_mg = block.evaluated_phi_mg
            assert phi_mg is not None
            start_c, end_c = block.get_domain_local_start_end_c()

            nt_g = nt_G[start_c[0]:end_c[0],
                        start_c[1]:end_c[1],
                        start_c[2]:end_c[2]]

            rho_mm = rho_MM[block.M_m][:, block.M_m]
            nt_g += np.einsum('mxyz, nxyz, mn -> xyz', phi_mg, phi_mg, rho_mm,
                              optimize=True)
