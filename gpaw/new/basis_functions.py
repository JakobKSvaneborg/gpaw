"""
Each atom has list of basis functions. A basis function is defined by its
angular quantum numbers l, m and a radial function f(r). We also require a
radial cutoff r_c so that f(r) = 0 for r >= r_c. Each basis function evaluates
to f(r) * Y_lm, r being the distance from the atom's center.

We use a combined index mu, or M, to enumerate basis functions of different
atoms and l,m: mu = {a, l, m}. In practice, allowed values of m depend on l so
we mainly use a and l when grouping basis functions. The radial function and
cutoff are the same for all m that share {a, l} indices. Mu is ordered such
that first are l=0 functions of atom a=0, then l=1 funcs of a=0, etc.

Class BasisFunctionCollection describes the full set of all basis functions
phi_mu and provides methods for calculating stuff with them.
"""

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from collections import defaultdict
from typing import TypeAlias, cast

import numpy as np
from gpaw.core import UGDesc
from gpaw.gpu import cupy as cp
from gpaw.new.timer import trace
from gpaw.setup import Setup, Setups
from gpaw.spline import Spline
from gpaw.typing import Array4D


BlockCoords: TypeAlias = tuple[int, int, int]


class PrecalculationMode(Enum):
    eIfPossible = auto()
    eAlways = auto()
    eNever = auto()


@dataclass
class LFCMatrixDistributionRules:
    """Used to describe how calculate_potential_matrix should be parallelized.
    We will parallelize by rows, ie. one rank does rows [mu_start, mu_end).
    """
    mu_start: int
    mu_end: int


@dataclass(frozen=True)
class BasisFunctionDesc:
    """Data for defining a basis function phi(x)."""

    l: int
    """Angular momentum quantum number."""
    cutoff: float
    """Cutoff for the radial part. phi(r) = 0 for r > cutoff"""
    f_r: np.ndarray
    """Discrete samples of the radial function f(r) on [0, cutoff].
    These values are used to construct a continuous spline. The
    final value at r = cutoff MUST be exactly zero."""

    def __repr__(self):
        """"""
        return f"BasisFunctionDesc(l={self.l}, cutoff={self.cutoff})"

    def __post_init__(self):
        """"""
        if self.l < 0:
            raise ValueError("l must be >= 0")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be > 0")
        if self.f_r.ndim != 1 or self.f_r.size < 3:
            raise ValueError(
                "Spline f_r needs to be 1D array with at 3 elements")
        if self.f_r[-1] != 0.0:
            raise ValueError("Last value in f_r must be 0")

    def __eq__(self, other: object) -> bool:
        """"""
        if not isinstance(other, BasisFunctionDesc):
            return NotImplemented
        return (
            self.l == other.l
            and self.cutoff == other.cutoff
            and np.array_equal(self.f_r, other.f_r)
        )

    @staticmethod
    def from_legacy_spline(spline: Spline) -> "BasisFunctionDesc":
        """"""
        breakpoints = np.linspace(0.0,
                                  spline.get_cutoff(),
                                  spline.get_npoints())
        f_r = spline.map(breakpoints)
        assert f_r[-1] == 0.0
        return BasisFunctionDesc(spline.get_angular_momentum_number(),
                                 spline.get_cutoff(),
                                 f_r)


@dataclass
class LFCSystemDesc:
    """Defines the LFC system: Grid to use and a list of atoms present in the
    main unit cell. Each atom must define a list of basis function
    descriptors.
    """
    grid: UGDesc
    phi_aj: list[list[BasisFunctionDesc]]
    relpos_ac: np.ndarray


class AtomStaticData:
    """Describes atom content of the system and their basis functions.
    Does not store dynamical info like atom positions.
    """

    def __init__(
            self,
            phi_aj: list[list[BasisFunctionDesc]]):
        """"""

        # Deduplicate desc -> spline index, using object identity
        id_to_spline_idx: dict[int, int] = {}
        num_uniques = 0
        unique_descs: list[BasisFunctionDesc] = []

        self.spline_indices_a: list[list[int]] = []
        """Gives spline indices for each basis function of given atom."""

        for phi_j in phi_aj:
            indices = []
            for desc in phi_j:
                desc_id = id(desc)
                if desc_id not in id_to_spline_idx:
                    id_to_spline_idx[desc_id] = num_uniques
                    unique_descs.append(desc)
                    num_uniques += 1

                indices.append(id_to_spline_idx[desc_id])

            self.spline_indices_a.append(indices)
            assert len(indices) > 0

        # Take deepcopies of the unique descs: we want exclusive ownership of
        # this data
        self.phi_i = [BasisFunctionDesc(desc.l, desc.cutoff, desc.f_r.copy())
                      for desc in unique_descs]
        """List of all unique phi descriptors in the system, ordered by their
        associated spline index."""

        self.phi_aj: list[list[BasisFunctionDesc]] = [
            [self.phi_i[idx] for idx in indices]
            for indices in self.spline_indices_a
        ]
        """Holds data-only descriptors of basis functions for each atom"""

        self.mu_range_a: list[range] = []
        """Range of mu indices for atom 'a'."""

        mu = 0
        for phi_j in self.phi_aj:
            mu_local = 0
            for phi in phi_j:
                mu_local += 2 * phi.l + 1

            self.mu_range_a.append(range(mu, mu + mu_local))
            mu += mu_local


@dataclass
class BasisFunctionInstance:
    """Instance of a basis function with fixed position and mu index range."""

    desc: BasisFunctionDesc
    """Phi metadata"""
    spline_index: int
    first_mu: int
    position: np.ndarray
    """Real-space position"""
    cell_coords: np.ndarray
    """Coords of the cell containing the center of this phi. 3D array of ints.
    These are relative to the main unit cell which has coords (0, 0, 0)."""

    def get_cutoff(self) -> float:
        """Get the radial cutoff"""
        return self.desc.cutoff

    def get_angular_momentum_number(self) -> int:
        """Get l"""
        return self.desc.l

    def get_num_mu(self) -> int:
        return 2 * self.desc.l + 1

    def find_overlapping_points(self, R_Gv: np.ndarray) -> np.ndarray:
        """"""
        assert R_Gv.ndim == 4
        diff = R_Gv - self.position
        distance = (np.sum(diff**2, axis=3))**0.5
        mask = (distance <= self.get_cutoff())
        return R_Gv[mask]


def build_lfc_system(setups: Setups,
                     grid: UGDesc,
                     relpos_ac: np.ndarray) -> LFCSystemDesc:
    """Old version of BasisFunctions takes just splines_aj: list[list[Spline]]
    as input, which is a list spanning all atoms. In practice, it's
    constructed as [setup.basis_functions_J for setup in setups].
    With just this it's hard to know which splines are shared between atoms.

    We also cannot directly use the existing splines as they do not support
    GPU or single-precision evaluation.

    This function reads spline data from the setups and puts it in format
    suitable for the new BasisFunctionCollection constructor.
    """

    num_atoms = len(setups)

    relpos_ac = np.asanyarray(relpos_ac)
    assert len(relpos_ac) == num_atoms

    """Setups generally have multiple copies of the same atom type
    => same splines appear many times. Here we loop over all atoms and
    identify those with unique spline lists, and only create phi descs for
    these splines. This can still result in duplicates in principle but the
    BasisFunctionCollection class filters those out later.
    """
    unique_spline_lists: list[list[Spline]] = []
    unique_phi_descs: list[list[BasisFunctionDesc]] = []
    phi_aj: list[list[BasisFunctionDesc]] = []

    for atom_idx, setup in enumerate(setups):
        setup = cast(Setup, setup)

        my_splines: list[Spline] = setup.basis_functions_J

        if my_splines not in unique_spline_lists:
            # Found new atom type
            unique_spline_lists.append(my_splines)

            phi_descs = []
            for spline in my_splines:
                phi_descs.append(BasisFunctionDesc.from_legacy_spline(spline))
            unique_phi_descs.append(phi_descs)

        else:
            idx = unique_spline_lists.index(my_splines)
            phi_descs = unique_phi_descs[idx]
        phi_aj.append(phi_descs)

    assert len(unique_phi_descs) == len(unique_spline_lists)

    return LFCSystemDesc(grid, phi_aj, relpos_ac)


@dataclass
class ImageSphereData:
    """"""
    position_iv: np.ndarray
    cell_shifts_ic: np.ndarray
    radius: float


class GeometryHelpers:
    """Geometry-related static data and helper functions. Most importantly
    this defines blocking of the grid domain into sub-grids of block_size_c
    grid points each. Blocks are labeled by integers (Bx, By, Bz), so that
    block (0, 0, 0) starts at grid.start_c and spans `block_size` grid points.
    Note that (Bx, By, Bz) are relative to the MPI domain. Last blocks in each
    direction can be smaller: their end_c are clipped so that the block does
    not extend outside the domain.

    Indexing:
        somearray_B => B refers to (Bx, By, Bz) block coords, ie. access as
        somearray_B[1, 2, 3].
    """

    def __init__(self, grid: UGDesc, block_size_c: np.ndarray):
        """Precompute and cache useful geometry-related data"""
        assert np.all(block_size_c) > 0
        assert np.all(block_size_c <= grid.mysize_c)

        self.grid = grid
        self.h_cv = (grid.cell_cv.T / grid.size_c).T
        """Grid spacing (precomputed; same as on the full grid)"""

        self.block_size_c = block_size_c

        num_blocks_c = -(self.grid.mysize_c // -self.block_size_c).astype(int)
        self.num_blocks_c = num_blocks_c
        """Number of blocks in this MPI domain in each grid direction."""

        # Compute grid domain AABB
        aabb_corners_c = np.array(list(itertools.product([0, 1], repeat=3)))
        span_c = self.grid.end_c - self.grid.start_c
        domain_aabb_corners_ic = (
            self.grid.start_c + aabb_corners_c * span_c)
        self.domain_aabb_corners_iv = domain_aabb_corners_ic @ self.h_cv
        """Real-space positions of AABB corners for our grid domain."""
        self.domain_aabb_min_v = self.domain_aabb_corners_iv.min(axis=0)
        """'Min' corner of the grid-domain AABB."""
        self.domain_aabb_max_v = self.domain_aabb_corners_iv.max(axis=0)
        """'Max' corner of the grid-domain AABB."""

        # Compute block geometry: start_c and end_c for each block.
        # end_c is clamped so that blocks don't extend outside the domain.
        # Block start/end in grid-index space: (Bx, By, Bz, 3)
        block_grid = np.stack(
            np.meshgrid(*[np.arange(n) for n in num_blocks_c], indexing='ij'),
            axis=-1)

        block_start_Bc = self.grid.start_c + self.block_size_c * block_grid
        block_end_Bc = np.minimum(block_start_Bc + self.block_size_c,
                                  self.grid.end_c)
        num_blocks_c = -(self.grid.mysize_c // -self.block_size_c).astype(int)

        self.block_start_Bc = block_start_Bc
        """Start grid indices of each block into the full grid"""
        self.block_end_Bc = block_end_Bc
        """End grid indices of each block into the full grid"""

        # AABB of each block: need all 8 corners since the cell can be skewed
        corner_offsets = np.array(list(itertools.product([0, 1], repeat=3)))

        block_corners_Bic = block_start_Bc[..., None, :] \
            + corner_offsets * (block_end_Bc - block_start_Bc)[..., None, :]

        self.block_corners_Biv = block_corners_Bic @ self.h_cv
        """Block corner positions in real space: (Bx, By, Bz, 8, 3)"""

    def get_grid_points_xyz(self, block: BlockCoords) -> Array4D:
        """Returns real-space XYZ coordinates for each grid point in the
        specified block. The block coords must be given relative to this MPI
        domain. See also grid.xyz()."""

        block_coords = np.asarray(block)
        if (np.any(block_coords < 0)
            or np.any(block_coords >= self.num_blocks_c)):
            raise ValueError("Invalid block coords (must be domain-relative)")

        start_c = self.block_start_Bc[block]
        end_c = self.block_end_Bc[block]
        indices_Rc = (
            np.indices(tuple(end_c - start_c)).transpose((1, 2, 3, 0)))
        indices_Rc += start_c
        return indices_Rc @ self.h_cv

    def find_image_spheres_with_cell_aabb(
        self,
        sphere_pos_v: np.ndarray,
        radius: float
    ) -> ImageSphereData:
        """Finds periodic images of the input sphere in other unit cells that
        overlap the main cell. This is a fast but overly conservative routine:
        the overlaps are computed using AABB of the main cell, which can
        result in false positives especially with non-orthorhombic cells. See
        other routines in GeometryHelpers for more precise culling of these
        false positives.
        """
        icell_cv = self.grid.icell.T
        h_c = 1.0 / np.linalg.norm(icell_cv, axis=0)

        # Sphere position in cell-fractional coords
        pos_c = sphere_pos_v @ icell_cv

        # How many cells can the sphere extend at max, in a given direction
        n_max_c = np.ceil(radius / h_c).astype(int) + 1

        # Handle non-periodic boundaries: no images in non-periodic dirs.
        # This matches the `cut=True` option in old BasisFunctions class;
        # the other option would throw GridBoundsError if some basis funcs
        # don't fit the main cell. But in basis.py the object was always
        # created with `cut=True`, so we only implement that version.
        n_max_c = np.where(self.grid.pbc_c, n_max_c, 0)

        # Build a 3D grid of sphere shifts. This Numpy magic is equivalent to:
        # for nx, ny, nz in itertools.product(
        #     range(-n_max_c[0], n_max_c[0] + 1),
        #     range(-n_max_c[1], n_max_c[1] + 1),
        #     range(-n_max_c[2], n_max_c[2] + 1),
        # ):
        #     shift = np.array([nx, ny, nz])
        #     ...
        ranges = [np.arange(-n, n + 1) for n in n_max_c]
        cells = np.meshgrid(*ranges, indexing="ij")
        shifts = np.stack(cells, axis=-1)
        # includes (0, 0, 0), ie. the main cell

        # Place image sphere candidates
        image_pos_ic = pos_c + shifts
        image_pos_iv = image_pos_ic @ self.grid.cell_cv

        # Flatten the output so that image_pos_iv[i] gives the position of
        # i.th sphere, and so on
        return ImageSphereData(
            image_pos_iv.reshape(-1, 3),
            shifts.reshape(-1, 3),
            radius)

    def cull_spheres_domain_aabb(
        self,
        spheres: ImageSphereData
    ) -> ImageSphereData:
        """Finds spheres that overlap the grid-domain bounding box."""
        closest_iv = np.clip(spheres.position_iv,
                             self.domain_aabb_min_v, self.domain_aabb_max_v)
        dist2_aabb = np.sum((spheres.position_iv - closest_iv)**2,
                            axis=-1)
        in_range = dist2_aabb <= spheres.radius**2

        return ImageSphereData(
            spheres.position_iv[in_range],
            spheres.cell_shifts_ic[in_range],
            spheres.radius)

    def find_overlapping_blocks(
        self,
        spheres: ImageSphereData
    ) -> tuple[ImageSphereData, dict[int, list[BlockCoords]],
               dict[BlockCoords, list[int]]]:
        """Finds which spheres overlap with which block in this grid domain.
        The output ImageSphereData contains just spheres that overlap with
        at least one block, and the two output dicts provide lookups for which
        spheres overlap with which blocks, and vice versa.

        This check is exact: it first does a fast block-AABB overlap check
        to cull distant spheres, then continues with a slower but exact
        per-grid-point computation for the surviving (sphere, block) pairs.
        """

        # Step 1: Cull distant spheres with block-AABB check (vectorized)
        aabb_min_Bv = self.block_corners_Biv.min(axis=-2)
        aabb_max_Bv = self.block_corners_Biv.max(axis=-2)

        # Let N = num_spheres

        r2 = spheres.radius**2
        pos_iv = spheres.position_iv
        # Broadcast sphere centers over block grid
        pos_iBv = pos_iv[:, None, None, None, :]  # (N, 1, 1, 1, 3)
        closest_iBv = np.clip(
            pos_iBv, aabb_min_Bv, aabb_max_Bv)  # (N, Bx, By, Bz, 3)
        dist2_iB = np.sum(
            (pos_iBv - closest_iBv)**2, axis=-1)      # (N, Bx, By, Bz)
        aabb_mask_iB = (dist2_iB <= spheres.radius**2)

        # Step 2: Calculate which spheres really overlap with which blocks.
        # Do this with an exact distance computation at each grid point

        exact_mask_iB = np.zeros_like(aabb_mask_iB)

        for bx, by, bz in zip(*np.where(aabb_mask_iB.any(axis=0))):
            # Mask spheres that survived the AABB culling (count: C):
            candidates = np.where(aabb_mask_iB[:, bx, by, bz])[0]   # (C,)
            points_Rv = self.get_grid_points_xyz(
                (int(bx), int(by), int(bz))).reshape(-1, 3)
            diff_CGv = pos_iv[candidates, None, :] - points_Rv[None, :, :]
            hits = np.sum(diff_CGv ** 2, axis=-1).min(axis=-1) <= r2  # (C,)
            exact_mask_iB[candidates[hits], bx, by, bz] = True

        # Final sphere culling
        sphere_has_overlap = exact_mask_iB.any(axis=(1, 2, 3))
        culled_mask_iB = exact_mask_iB[sphere_has_overlap]

        culled_spheres = ImageSphereData(
            spheres.position_iv[sphere_has_overlap],
            spheres.cell_shifts_ic[sphere_has_overlap],
            spheres.radius)

        # Build sphere <-> block lookups
        sphere_idx_to_blocks: dict[int, list[BlockCoords]] = defaultdict(list)
        block_to_sphere_indices: dict[BlockCoords, list[int]] \
            = defaultdict(list)

        for new_i, bx, by, bz in zip(*np.where(culled_mask_iB)):
            block = (int(bx), int(by), int(bz))
            sphere_idx_to_blocks[int(new_i)].append(block)
            block_to_sphere_indices[block].append(int(new_i))

        return culled_spheres, sphere_idx_to_blocks, block_to_sphere_indices


class BasisFunctionCollectionBase(ABC):
    """Base class for LFC basis functions (new implementation). Handles atom
    and phi_mu placement and their index ordering, and blocking of the grid.
    Also acts as a "control" class for orchestrating LFC computations.
    Subclasses can implement more efficient data structures etc for their
    specific implementations (or let a C++ side object do it).

    Access to splines is via a unique 'spline index' integer and eg. phi
    instances store this ID instead of a spline reference directly. This is
    done to have full separation of spline lookups and spline implementations.
    """

    _matrix_distribution_rules: LFCMatrixDistributionRules
    """FIXME: why does this have to be stored?! Old code passes the M ranges
    to C functions anyway. Outside users do seems to access Mstart, Mend,
    why?!?"""

    def __init__(self,
                 system: LFCSystemDesc,
                 use_gpu: bool = False,
                 mode: PrecalculationMode = PrecalculationMode.eIfPossible,
                 block_size: int | tuple[int, int, int] | None = 8):
        """
        Initialize a BasisFunctionCollection.

        Parameters
        ----------
        system : LFCSystemDesc
            Defines the grid and atom content on the grid. Should contain all
            atoms in the main unit cell even if the grid is set to use domain
            decomposition.
        use_gpu : bool, optional
            If True, basis functions will be computed and stored
            **exclusively** on GPU.
            Default is False.
        mode : PrecalculationMode, optional
            Control whether basis functions should be precalculated on the
            grid. Precalculation occurs whenever atom positions are changed.
            Warning: For realistic systems this may require extreme
            amounts of memory!
            Default is eIfPossible (precalculate if memory allows).
        block_size: int | tuple[int, int, int], optional
            Size of each grid block. None means no blocking.
        """
        self.grid = system.grid
        if len(system.phi_aj) == 0:
            raise RuntimeError("No atoms?!")

        self.xp = cp if use_gpu else np

        if block_size is not None:
            real_block_size = np.asanyarray(block_size)
            real_block_size = np.minimum(real_block_size, self.grid.mysize_c)
        else:
            real_block_size = self.grid.mysize_c.copy().astype(int)
        real_block_size = real_block_size.astype(int)

        # TODO let caller choose dtype (32 or 64bit float)
        self.dtype = np.float64

        self.geometry = GeometryHelpers(self.grid, real_block_size)
        """Contains static data about block geometry"""

        # TODO: actually check if we can/should precalculate
        if mode != PrecalculationMode.eNever:
            # if self.get_cache_size_estimate() > too_much...
            self.should_precalculate = True
        else:
            self.should_precalculate = False

        self.atom_static_data = AtomStaticData(system.phi_aj)
        """Static data about the atom contents, including lookups based on mu
        (global phi index) and spline indexing."""

        self._init_splines(self.atom_static_data.phi_i)

        # Default is to always compute the full V_munu matrix
        self.set_matrix_distribution(0, self.Mmax)

        self._relpos_ac: np.ndarray = system.relpos_ac
        """Current grid-relative positions atoms."""

        self._rank_a = [-1] * self.num_atoms
        """Which MPI grid domain does each atom currently reside in."""

        self.phi_instances_ai: dict[int, list[BasisFunctionInstance]] \
            = defaultdict(list)
        """Maps atom index -> list of all basis function instances of that atom
        ("phi spheres") that currently overlap our grid domain. Will differ for
        each MPI rank. The phi list can be empty."""

        self._block_overlaps_a: \
            list[dict[BlockCoords, list[BasisFunctionInstance]]] \
            = [{}] * self.num_atoms
        """List of maps for each atom: block -> overlapping phi instance."""

        # this triggers phi instance rebuild and updates block lookups:
        self.set_positions(self._relpos_ac, force_update=True)

    # Begin abstracts
    @trace
    @abstractmethod
    def precalculate_on_grid(self, changed_atoms_a: np.ndarray) -> None:
        """Implements basis function precalculation in all grid points where
        they contribute. The input array changed_atoms_a specifies which atoms
        have changed since the previous precalculation."""
        raise NotImplementedError

    @trace
    @abstractmethod
    def add_to_density(
            self,
            nt_sG: np.ndarray | cp.ndarray,
            f_asi: dict[int, np.ndarray] | dict[int, cp.ndarray]) -> None:
        r"""Add linear combination of squared localized functions to density.
        Note that different atoms can have different number of basis funcs, so
        the range of `i` can vary. Therefore, f_asi is a dict (or a list) of
        arrays, not an array itself.
        :::

          ~        ---   a    a   2
          n (r) += >    f   [Φ(r)]
           s       ---   si   i
                   a,i
        """
        raise NotImplementedError

    @trace
    @abstractmethod
    def calculate_potential_matrix(
            self,
            vt_G: np.ndarray | cp.ndarray,
            out: np.ndarray | cp.ndarray | None = None) \
            -> np.ndarray | cp.ndarray:
        """Calculate lower part of the potential matrix.

        ::

                      /
            ~         |     *  _  ~ _        _   _
            V      =  |  Phi  (r) v(r) Phi  (r) dr    for  mu >= nu
             mu nu    |     mu            nu
                      /

        Parameters
        ----------
        vt_G : np.ndarray | cp.ndarray
            The potential array in real space (GPU or CPU).
        out : np.ndarray | cp.ndarray | None, optional
            Optional output array to store the result. If None, a new array is
            created. Default is None. TODO shape requirements?

        Returns
        -------
        np.ndarray | cp.ndarray
            The potential matrix with shape (Mstop - Mstart, MMax), where:
            - Rows correspond to basis functions [Mstart, Mstop)
            (if set_matrix_distribution was called, otherwise all rows).
            - Only the lower triangle and diagonal are guaranteed to be
            correct. Upper triangle elements may contain undefined or stale
            values.

            If 'out' was given, return value will be that same array.

        Notes
        -----
        - Only the lower triangle and diagonal elements are guaranteed to be
        correct. Upper triangle may still be modified (for example, the old
        CPU code does this).
        - If set_matrix_distribution() has been called, only rows
        [Mstart, Mstop) are computed.
        - The returned array type (CPU or GPU) matches the input vt_G type.
        """

        raise NotImplementedError

    @abstractmethod
    def construct_density(self,
                          rho_MM,
                          nt_G: np.ndarray | cp.ndarray,
                          q):
        raise NotImplementedError

    @abstractmethod
    def has_precalculated_phi(self) -> bool:
        """"""
        raise NotImplementedError

    @abstractmethod
    def _init_splines(self, phi_i: list[BasisFunctionDesc]) -> None:
        """Creates concrete spline objects for each phi descriptor in the
        input list."""
        raise NotImplementedError
    # End abstracts

    def calculate_potential_matrices(
            self,
            vt_G: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """"""
        V_MM = self.calculate_potential_matrix(vt_G)
        return V_MM[np.newaxis]

    def set_matrix_distribution(self, Mstart: int, Mstop: int) -> None:
        """Specifies that calculate_potential_matrices should only do rows
        [Mstart, Mstop); Mstop is exclusive. Used for parallelization.
        See dataclass LFCMatrixDistributionRules."""
        # FIXME: why do we need stateful Mstart, Mstop?
        # Why not just pass them as inputs to functions that use them?

        Mstart = max(Mstart, 0)
        Mstop = min(Mstop, self.Mmax)
        self._matrix_distribution_rules \
            = LFCMatrixDistributionRules(Mstart, Mstop)

    @cached_property
    def Mmax(self) -> int:
        """Last global index mu (exclusive!)."""
        return self.get_mu_range_a(self.num_atoms - 1).stop

    @property
    def mu_range(self) -> range:
        """Allowed range of mu indices"""
        return range(0, self.Mmax)

    def get_mu_range_a(self, atom_index: int) -> range:
        """Get range of mu indices for specified atom."""
        return self.atom_static_data.mu_range_a[atom_index]

    def get_num_phi_a(self, atom_index: int) -> int:
        """Get number of basis functions for specified atom"""
        mu_range = self.get_mu_range_a(atom_index)
        return mu_range.stop - mu_range.start

    def get_phi_instances(self) -> list[BasisFunctionInstance]:
        """Returns all basis function instances that contribute in this unit
        cell, including periodic copies extending from other cells"""
        phi_i = []
        for phi_j in self.phi_instances_ai.values():
            phi_i += phi_j
        return phi_i

    def get_phi_instances_for_atom(
            self,
            atom_idx: int) -> list[BasisFunctionInstance]:
        """Returns list of current basis function instances originating from
        given atom. Only includes basis funcs that overlap with the current
        grid domain."""
        return self.phi_instances_ai[atom_idx]

    def get_nonempty_blocks(self) -> list[BlockCoords]:
        """Returns list of block coordinates (Bx, By, Bz) of blocks that have
        overlap with at least one basis function. CAUTION: slow!"""
        blocks = []
        for block_to_phi in self._block_overlaps_a:
            for block, phi_j in block_to_phi.items():
                if len(phi_j) > 0:
                    blocks.append(block)

        return blocks

    def get_block_to_phi_map(self) \
            -> dict[BlockCoords, list[BasisFunctionInstance]]:
        """Returns map: block (Bx, By, Bz) -> list of basis function instances
        that overlap that block. CAUTION: slow!"""
        block_to_phi = defaultdict(list)
        for block_to_phi_atom in self._block_overlaps_a:
            for block, phi_j in block_to_phi_atom.items():
                block_to_phi[block].extend(phi_j)

        return block_to_phi

    # ??? These need to be exposed directly, outside code accesses them...
    @property
    def Mstart(self) -> int:
        """"""
        return self._matrix_distribution_rules.mu_start

    @property
    def Mstop(self) -> int:
        """"""
        return self._matrix_distribution_rules.mu_end
    # end ???

    def uses_gpu(self) -> bool:
        """Returns True if the basis functions and splines are allocated in
        GPU memory."""
        return self.xp is not np

    # TODO:
    # def get_cache_size_estimate(self) -> int:
    #     """Estimate of how many bytes are needed to precalculate and cache
    #     the basis functions on the grid.
    #     """
    #     num_sites = 0
    #     for block in self.get_relevant_blocks():
    #         num_sites += int(np.prod(block.shape))

    #     dummy = np.empty((1), dtype=self.dtype)
    #     return num_sites * self.Mmax * dummy.itemsize

    @property
    def num_atoms(self) -> int:
        """Total number of atoms in the system (not just in this MPI rank)."""
        return len(self.atom_static_data.phi_aj)

    @cached_property
    def atom_indices(self) -> list[int]:
        """Returns list of all atom indices in the system.
        See also my_atom_indices."""
        return list(range(0, self.num_atoms))

    @property
    def my_atom_indices(self) -> list[int]:
        """Indices of atoms in this MPI domain"""
        myrank = self.grid.comm.rank
        return [a for a, r in enumerate(self._rank_a) if r == myrank]

    def get_atom_positions(self, grid_relative=False) -> np.ndarray:
        """Returns current atom positions. Either position_av or relpos_ac."""
        if grid_relative:
            return self._relpos_ac
        else:
            return self._relpos_ac @ self.grid.cell_cv

    @dataclass
    class AtomUpdateResult:
        relpos_c: np.ndarray
        """Relative position of the atom after move"""
        rank: int
        """MPI rank of the atom center after move"""
        phi_spheres: list[BasisFunctionInstance]
        """Phi spheres for this atom that now overlap the main cell in this
        MPI domain. Includes periodic copies of phi in other cells."""
        block_to_phi: dict[BlockCoords, list[BasisFunctionInstance]]
        """Which blocks now overlap with which phi spheres of this atom"""

    @trace
    def set_positions(self, relpos_ac: np.ndarray, force_update: bool) -> bool:
        """Updates atom positions. Return value is True if any atom migrated
        to another MPI rank in grid domain decomposition.
        If force_update is true, does a full init without caring about
        existing state.
        This is a rather expensive routine! Internally it goes roughly as
        follows:
        1. For each moved atom and each basis function associated with it, a
        fast but conservative overlap calculation is performed to find which
        periodic copies of phi from other cells overlap with the main cell.
        2. The overlap results from step 1 are refined and made exact for each
        phi by finding grid points that they overlap. This utilizes grid
        blocking.
        3. Lookup arrays are constructed for mapping block indices -> list of
        phi instances that overlap the block.

        Note that the total number of contributing phi instances may change
        every time atoms move, due to changes in periodic overlaps.
        """
        relpos_ac = np.asanyarray(relpos_ac)

        if len(relpos_ac) != self.num_atoms or relpos_ac.ndim != 2 \
                or relpos_ac.shape[1] != 3:
            raise ValueError("Wrong atom position array shape")

        has_migrations = False
        has_changes_a = np.zeros((self.num_atoms), dtype=bool)
        old_relpos_ac = self.get_atom_positions(True)

        for a in self.atom_indices:

            new_relpos = relpos_ac[a]
            if np.array_equal(old_relpos_ac, new_relpos) and not force_update:
                # Nothing to do for this atom
                continue

            res = self._update_atom_position(a, new_relpos)
            has_changes_a[a] = True

            if self._rank_a[a] != res.rank:
                has_migrations = True
                self._rank_a[a] = res.rank

            self.phi_instances_ai[a] = res.phi_spheres
            # Update block lookups for the atom
            self._block_overlaps_a[a] = res.block_to_phi

        self._relpos_ac = relpos_ac

        if self.should_precalculate and np.any(has_changes_a):
            self.precalculate_on_grid(has_changes_a)

        return has_migrations

    def _update_atom_position(
            self,
            atom_idx: int,
            new_relpos_c: np.ndarray) -> AtomUpdateResult:
        """"""
        new_rank = self.grid._gd.get_rank_from_position(new_relpos_c)

        # Handle periodic boundaries: basis funcs from other unit cells
        # can extend to the "main" cell. Treat this as a problem of N
        # spheres (phi) in a box, and we want to know which spheres overlap
        # with a smaller box in the middle ("main" cell). We first have
        # to construct the "periodic images" of spheres, then perform
        # sphere-box overlap checks.

        position_v = new_relpos_c @ self.grid.cell_cv

        spline_indices_j = self.atom_static_data.spline_indices_a[atom_idx]
        mu = self.atom_static_data.mu_range_a[atom_idx].start

        # Accumulate info about all phi instances originating from this atom
        phi_i: list[BasisFunctionInstance] = []
        block_to_phi: dict[BlockCoords, list[BasisFunctionInstance]] \
            = defaultdict(list)

        for j, phi in enumerate(self.atom_static_data.phi_aj[atom_idx]):

            spline_idx = spline_indices_j[j]
            radius = phi.cutoff
            image_spheres = self.geometry.find_image_spheres_with_cell_aabb(
                position_v,
                radius)

            # --- Fast pass: sphere vs AABB of the grid domain ---
            image_spheres = self.geometry.cull_spheres_domain_aabb(
                image_spheres
            )

            # --- Exact pass: sphere vs grid block points ---
            image_spheres, sphere_idx_to_blocks, block_to_sphere_indices \
                = self.geometry.find_overlapping_blocks(image_spheres)

            for i in range(len(image_spheres.position_iv)):
                new_phi = BasisFunctionInstance(
                    phi,
                    spline_idx,
                    mu,
                    image_spheres.position_iv[i],
                    image_spheres.cell_shifts_ic[i])

                phi_i.append(new_phi)

                # Accumulate block lookup
                for block in sphere_idx_to_blocks[i]:
                    block_to_phi[block].append(new_phi)

            mu += (2 * phi.l + 1)

        return self.AtomUpdateResult(
            new_relpos_c,
            new_rank,
            phi_i,
            block_to_phi)
