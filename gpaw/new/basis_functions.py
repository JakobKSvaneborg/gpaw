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
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from typing import TypeAlias, cast

import numpy as np
from gpaw.core import UGDesc
from gpaw.gpu import cupy as cp
from gpaw.new.timer import trace
from gpaw.setup import Setup, Setups
from gpaw.spline import Spline
from gpaw.typing import Array4D


BlockCoords: TypeAlias = tuple[int, int, int]


def find_image_spheres_conservative(
        grid: UGDesc,
        sphere_pos_v: np.ndarray,
        radius: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """"""
    icell_cv = grid.icell.T
    h_c = 1.0 / np.linalg.norm(icell_cv, axis=0)

    # Sphere position in cell-fractional coords
    pos_c = sphere_pos_v @ icell_cv

    # How many cells can the sphere extend at max, in a given direction
    n_max_c = np.ceil(radius / h_c).astype(int) + 1

    # Handle non-periodic boundaries: can't have images in non-periodic dirs.
    # This matches the `cut=True` option in old BasisFunctions class;
    # the other option would throw GridBoundsError if some basis funcs don't
    # fit the main cell. But in basis.py the object was always created with
    # `cut=True`, so we only implement that version.
    n_max_c = np.where(grid.pbc_c, n_max_c, 0)

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
    image_pos_c = pos_c + shifts
    image_pos_v = image_pos_c @ grid.cell_cv

    # --- Fast conservative pass: sphere vs AABB of the main cell ---
    aabb_corners_c = np.array(list(itertools.product([0, 1], repeat=3)))
    aabb_corners_v = aabb_corners_c @ grid.cell_cv
    aabb_min = aabb_corners_v.min(axis=0)
    aabb_max = aabb_corners_v.max(axis=0)

    closest_v = np.clip(image_pos_v, aabb_min, aabb_max)
    dist2_aabb = np.sum((image_pos_v - closest_v)**2, axis=-1)
    in_range = dist2_aabb < radius**2
    candidate_shifts = shifts[in_range]

    return list(candidate_shifts), list(image_pos_v[in_range])


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


@dataclass
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
            return False
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
class LFCAtomDesc:
    """Used in LFC creation, see LFCSystemDesc."""
    phi_j: list[BasisFunctionDesc]
    """List of basis functions for this atom."""
    position: np.ndarray
    """Real-space position of the atom. 3D array."""


@dataclass
class LFCSystemDesc:
    """Defines the LFC system: Grid to use and a list of atoms present in the
    main unit cell. Each atom should list its position and give a data-only
    description of its basis functions."""
    grid: UGDesc
    atoms: list[LFCAtomDesc]


@dataclass
class GridBlock:
    """Blocks are identified by their 3D indices on a "grid of blocks", ie.
    block (0, 0, 0) starts at grid.start_c and extends BLOCK_SIZE points
    in xyz dirs. Block (1, 0, 0) starts at
        grid.start_c + BLOCK_SIZE * [1, 0, 0],
    and so on.

    If the MPI domain does not divide evenly into blocks, we allow the last
    blocks in each direction to be smaller as to not "overflow" into the next
    domain. Note that with this we can also get non-box shaped blocks, eg.
    2D slab can be a valid block shape."""

    grid: UGDesc
    coords: BlockCoords

    # these give the range of grid points this block represents on the full
    # grid, NOT relative to the MPI domain
    start_c: np.ndarray
    """Start indices to the full (x,y,z) grid for this block. Inclusive."""
    end_c: np.ndarray
    """End indices to the full (x,y,z) grid for this block. Exclusive!"""
    phi_j: list["BasisFunctionInstance"] = field(default_factory=list)
    """List of all basis functions that have overlap with this block.
    Note indexing: each phi_j actually contains many values of m.
    On periodic systems, this includes basis funcs from unit cells that extend
    to this cell."""
    M_m: list[int] = field(default_factory=list)
    """Maps block-local phi index 'm' to global mu"""
    evaluated_phi_mg: np.ndarray | None = None
    """phi_mu(x) precalculated on the grid block. NOTE: the XYZ shape can be
    smaller for some blocks if the grid blocking was uneven.
    See also get_block_shape()"""
    phi_idx_j: list[int] = field(default_factory=list)
    """Maps block-local basis function index to global phi index"""

    def __post_init__(self):
        """"""
        assert all(x >= 0 for x in self.coords), \
            "Block coords are relative to MPI domain and must be >= 0"

        # end_c is exclusive so the following is OK even for non-3D shapes:
        assert np.all(self.end_c > self.start_c)
        assert np.all(self.start_c >= self.grid.start_c)
        assert np.all(self.end_c <= self.grid.end_c)

    def get_block_xyz(self) -> np.ndarray:
        """Gives real-space XYZ points for the given block.
        4D array of shape (Nx, Ny, Nz, 3)."""
        # See UGDesc.xyz()
        indices_Rc = np.indices(tuple(self.shape)).transpose((1, 2, 3, 0))
        indices_Rc += self.start_c
        return indices_Rc @ (self.grid.cell_cv.T / self.grid.size_c).T

    def get_local_start_end_c(self) -> tuple[np.ndarray, np.ndarray]:
        """start_c and end_c for looping over block-local XYZ arrays."""
        local_start_c = np.asarray([0, 0, 0])
        local_end_c = self.end_c - self.start_c
        return local_start_c, local_end_c

    def get_domain_local_start_end_c(self) -> tuple[np.ndarray, np.ndarray]:
        """Get start_c and end_c for this block relative to the grid domain.
        These are safe to use when slicing or looping over domain-distributed
        arrays.
        """
        domain_start_c = self.start_c - self.grid.start_c
        domain_end_c = domain_start_c + self.shape
        return domain_start_c, domain_end_c

    @cached_property
    def shape(self) -> tuple[int, int, int]:
        """Real shape of the block, ie. how many grid points it contains.
        This may in some situations be a 2D shape, but always contains at
        least one point."""
        return tuple(self.end_c - self.start_c)

    def get_block_shape(self) -> tuple[int, int, int]:
        """Gives real shape of the block."""
        return self.shape


@dataclass
class BasisFunctionInstance:
    """Instance of a basis function with fixed position and index range."""
    index: int
    """Global ID for this phi."""
    desc: BasisFunctionDesc
    """Phi metadata"""
    spline_index: int
    parent_atom: int
    first_mu: int
    position: np.ndarray
    """Real-space position"""
    cell_coords: np.ndarray
    """Coords of the cell containing the center of this phi. 3D array of ints.
    These are relative to the main unit cell which has coords (0, 0, 0)."""

    blocks_with_overlap: list[GridBlock] = field(default_factory=list)

    def __repr__(self) -> str:
        """Make this class printable. Default __repr__ would do infinite
        recursion into blocks_with_overlap. Change that to only show block
        count."""
        return (
            f"BasisFunctionInstance("
            f"index={self.index}, "
            f"desc={self.desc!r}, "
            f"spline_index={self.spline_index}, "
            f"parent_atom={self.parent_atom}, "
            f"first_mu={self.first_mu}, "
            f"position={self.position}, "
            f"blocks_with_overlap=<{len(self.blocks_with_overlap)} blocks>"
            f")"
        )

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

    atoms = []
    num_atoms = len(setups)

    relpos_ac = np.asanyarray(relpos_ac)
    assert len(relpos_ac) == num_atoms

    # atom positions in real units
    pos_av = relpos_ac @ grid.cell_cv

    """Setups generally have multiple copies of the same atom type
    => same splines appear many times. Here we loop over all atoms and
    identify those with unique spline lists, and only create phi descs for
    these splines. This can still result in duplicates in principle but the
    BasisFunctionCollection class filters those out later.
    """
    unique_spline_lists: list[list[Spline]] = []
    unique_phi_descs: list[list[BasisFunctionDesc]] = []
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

        atoms.append(LFCAtomDesc(phi_descs, pos_av[atom_idx]))

    assert len(atoms) == num_atoms
    assert len(unique_phi_descs) == len(unique_spline_lists)

    return LFCSystemDesc(grid, atoms)


@dataclass
class PhiSphereData:
    """Atom position updates produce these, gets turned to
    BasisFunctionInstance later. Reason for this separation is that
    BasisFunctionInstance needs a unique ID that should be assigned by
    the control LFC class and trying to assign these during position update
    is error prone."""
    pos_v: np.ndarray
    spline_idx: int
    parent_atom_idx: int
    first_mu: int
    phi_desc: BasisFunctionDesc
    cell_coords: np.ndarray
    """Length-3 array of int. Coords of the cell that this sphere resides in.
    """


class SplinePoolBase(ABC):
    """"""
    def __init__(self):
        """"""
        self.splines = []

    @abstractmethod
    def add_spline(self, desc: BasisFunctionDesc) -> None:
        """"""
        raise NotImplementedError

    def num_splines(self) -> int:
        """"""
        return len(self.splines)

    def get_spline(self, spline_idx: int):
        """"""
        return self.splines[spline_idx]


class BlockData:
    """Describes blocking of the grid domain into sub-grids of block_size_c
    grid points each. Blocks are labeled by integers (Bx, By, Bz), so that
    block (0, 0, 0) starts at grid.start_c and spans `block_size` grid points.
    Note that (Bx, By, Bz) are relative to the MPI domain. Last blocks in each
    direction can be smaller: their end_c are clipped so that the block does
    not extend outside the domain.
    """

    def __init__(self, grid: UGDesc, block_size_c: np.ndarray):
        """"""
        assert np.all(block_size_c) > 0
        assert np.all(block_size_c <= grid.mysize_c)

        self.grid = grid
        self.block_size_c = block_size_c

        num_blocks_c = -(self.grid.mysize_c // -self.block_size_c).astype(int)
        self.num_blocks_c = num_blocks_c
        """Number of blocks in this MPI domain in each grid direction."""

        # Compute block geometry: start_c and end_c for each block.
        # end_c is clamped so that blocks don't extend outside the domain.
        # Block start/end in grid-index space: (Bx, By, Bz, 3)
        block_grid = np.stack(
            np.meshgrid(*[np.arange(n) for n in num_blocks_c], indexing='ij'),
            axis=-1)

        block_start_c = self.grid.start_c + self.block_size_c * block_grid
        block_end_c = np.minimum(block_start_c + self.block_size_c,
                                 self.grid.end_c)
        num_blocks_c = -(self.grid.mysize_c // -self.block_size_c).astype(int)

        self.block_start_c = block_start_c
        """Start grid indices of each block into the full grid"""
        self.block_end_c = block_end_c
        """End grid indices of each block into the full grid"""

        # AABB of each block: need all 8 corners since the cell can be skewed
        corner_offsets = np.array(list(itertools.product([0, 1], repeat=3)))

        block_corners_c = block_start_c[..., None, :] \
            + corner_offsets * (block_end_c - block_start_c)[..., None, :]

        self.h_cv = (grid.cell_cv.T / grid.size_c).T
        """Grid spacing (precomputed; same as on the full grid)"""

        self.block_corners_v = block_corners_c @ self.h_cv
        """Block corner positions in real space: (Bx, By, Bz, 8, 3)"""

    def get_grid_points_xyz(self, block: BlockCoords) -> Array4D:
        """Returns real-space XYZ coordinates for each grid point in the
        specified block. The block coords must be given relative to this MPI
        domain. See also grid.xyz()."""

        block_coords = np.asarray(block)
        if (np.any(block_coords < 0)
            or np.any(block_coords >= self.num_blocks_c)):
            raise ValueError("Invalid block coords (must be domain-relative)")

        start_c = self.block_start_c[*block]
        end_c = self.block_end_c[*block]
        indices_Rc = (
            np.indices(tuple(end_c - start_c)).transpose((1, 2, 3, 0)))
        indices_Rc += start_c
        return indices_Rc @ self.h_cv


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

    phi_datas: dict[int, BasisFunctionDesc]
    """Holds definitions of all basis functions _types_ in the system,
    assigning a unique integer ID to each (the dict key). This is the single
    source of truth for indexing splines: Subclasses should construct their
    splines based on this data and use the same IDs for splines."""

    spline_indices_for_a: dict[int, list[int]]
    """Maps atom index => list of spline indices for that atom. This is
    constructed very early on to avoid lookup loops later."""

    spline_pool: SplinePoolBase
    """Holds the actual splines"""

    phi_i: list[BasisFunctionInstance]
    """List of all basis function instances in the system ("spheres").
    This includes periodic copies of the phi from neighboring cells, if those
    overlap the main cell."""

    _mu_range_a: list[range]
    """Range of mu indices for atom 'a'."""

    _rank_a: list[int]
    """MPI rank of the grid domain for specified atom index."""

    _relpos_ac: np.ndarray
    """Grid-relative positions atoms."""

    _matrix_distribution_rules: LFCMatrixDistributionRules
    """FIXME: why does this have to be stored?! Old code passes the M ranges
    to C functions anyway. Outside users do seems to access Mstart, Mend,
    why?!?"""

    blocks: list[GridBlock]
    # Store just those blocks that actually have phi overlap

    block_data: BlockData
    """Contains static data about block geometry"""

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
        if len(system.atoms) == 0:
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

        self.block_data = BlockData(self.grid, real_block_size)

        # TODO: actually check if we can/should precalculate
        if mode != PrecalculationMode.eNever:
            # if self.get_cache_size_estimate() > too_much...
            self.should_precalculate = True
        else:
            self.should_precalculate = False

        position_av = np.empty((len(system.atoms), 3))
        for a, atom in enumerate(system.atoms):
            position_av[a] = atom.position

        self._relpos_ac = position_av @ np.linalg.inv(self.grid.cell_cv)

        self._init_phi_datas(system.atoms)
        self._init_splines()
        self._init_mu_lookups()

        # set_positions trigger phi instance rebuild and block rebuild
        self._rank_a = [-1] * self.num_atoms
        self.set_positions(self._relpos_ac, force_update=True)

        # Default is to always compute do the full V_munu matrix
        self.set_matrix_distribution(0, self.Mmax)

    # Begin abstracts
    @trace
    @abstractmethod
    def precalculate_on_grid(self) -> None:
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
    def _init_splines(self) -> None:
        """Creates and sets self.spline_pool"""
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

        Mstart = max(Mstart, self.mu_range.start)
        Mstop = min(Mstop, self.mu_range.stop)
        self._matrix_distribution_rules \
            = LFCMatrixDistributionRules(Mstart, Mstop)

    def get_relevant_blocks(self) -> list[GridBlock]:
        """Returns list of grid blocks that contribute, ie. have overlap with
        some phi."""
        return self.blocks

    @cached_property
    def mu_range(self) -> range:
        """Range of mu indices"""
        # atom indices for lookups are always in range 0, ... N
        mu_start = self._mu_range_a[0].start
        mu_end = self._mu_range_a[-1].stop
        return range(mu_start, mu_end)

    @property
    def Mmax(self) -> int:
        """Last global index mu."""
        return self.mu_range.stop

    def get_mu_range_a(self, atom_index: int) -> range:
        """Get range of mu indices for specified atom."""
        return self._mu_range_a[atom_index]

    def get_num_phi_a(self, atom_index: int) -> int:
        """Get number of basis functions for specified atom"""
        mu_range = self._mu_range_a[atom_index]
        return mu_range.stop - mu_range.start

    def num_basis_functions(self) -> int:
        """Returns the total number of mu indices."""
        return self.mu_range.stop - self.mu_range.start

    def get_phi_instances(self) -> list[BasisFunctionInstance]:
        """Returns all basis function instances that contribute in this unit
        cell, including periodic copies extending from other cells"""
        return self.phi_i

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

    def get_cache_size_estimate(self) -> int:
        """Estimate of how many bytes are needed to precalculate and cache the
        basis functions on the grid.
        """
        num_sites = 0
        for block in self.get_relevant_blocks():
            num_sites += int(np.prod(block.shape))

        dummy = np.empty((1), dtype=self.dtype)
        return num_sites * self.num_basis_functions() * dummy.itemsize

    def get_phi_data_for_atom(self, atom_idx: int) -> list[BasisFunctionDesc]:
        """"""
        out = []
        for spline_idx in self.spline_indices_for_a[atom_idx]:
            out.append(self.phi_datas[spline_idx])
        return out

    def get_spline(self, spline_idx: int):
        """"""
        return self.spline_pool.get_spline(spline_idx)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms in the system (not just in this MPI rank)."""
        return len(self.spline_indices_for_a.keys())

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

    def _init_phi_datas(self, atoms: list[LFCAtomDesc]) -> None:
        """Setups self.phi_datas and self.spline_indices_for_a lookups."""
        self.phi_datas = {}
        self.spline_indices_for_a = {}
        num_unique_phi = 0

        for a, atom in enumerate(atoms):
            self.spline_indices_for_a[a] = []
            for phi in atom.phi_j:

                # avoid creating duplicate phi data (OK in principle)
                existing_spline_idx = -1
                for spline_idx, phi_data in self.phi_datas.items():
                    if phi is phi_data:
                        existing_spline_idx = spline_idx
                        break

                if existing_spline_idx >= 0:
                    self.spline_indices_for_a[a].append(existing_spline_idx)
                else:
                    # Found new phi
                    self.phi_datas[num_unique_phi] = phi
                    self.spline_indices_for_a[a].append(num_unique_phi)
                    num_unique_phi += 1

        # Take deepcopies of the phi descs to prevent surprises (LFC is
        # supposed to own its phi data)
        for k, v in self.phi_datas.items():
            phi_copy = BasisFunctionDesc(v.l, v.cutoff, v.f_r.copy())
            self.phi_datas[k] = phi_copy

        assert len(self.phi_datas) > 0

    def _init_mu_lookups(self) -> None:
        """This fixes how atom index is related to the global mu index."""
        self._mu_range_a = []
        mu = 0
        for a in range(self.num_atoms):

            mu_local = 0
            for phi in self.get_phi_data_for_atom(a):
                mu_local += 2 * phi.l + 1

            self._mu_range_a.append(range(mu, mu + mu_local))
            mu += mu_local

    def _update_block_data(self, spheres_i: list[PhiSphereData]) -> None:
        """"""

        phi_idx = 0
        self.phi_i = []
        block_map: dict[BlockCoords, GridBlock] = {}
        # Fast but conservative AABB pass: cull distant block-sphere pairs

        # shape (Bx, By, Bz, 3)
        block_aabb_min = self.block_data.block_corners_v.min(axis=-2)
        block_aabb_max = self.block_data.block_corners_v.max(axis=-2)

        # Gather sphere data into arrays.
        # Let S = len(spheres_i)
        pos_iv = np.array([s.pos_v for s in spheres_i])  # (S, 3)
        radius_i = np.array([s.phi_desc.cutoff for s in spheres_i])  # (S,)
        # For array broadcasting
        pos_iBv = pos_iv[:, None, None, None, :]
        radius_iB = radius_i[:, None, None, None]

        # Vectorized box-sphere overlap check for all pairs
        closest_iBv = np.clip(pos_iBv, block_aabb_min, block_aabb_max)
        dist2_iB = np.sum((closest_iBv - pos_iBv)**2, axis=-1)
        aabb_overlap_mask = (dist2_iB <= radius_iB**2)  # (S, Bx, By, Bz)

        # FIXME clearly we don't need the phi sphere struct to be separate
        # from BasisFunctionInstance. We should have some flattened PhiData
        # array that handles phi indexing, so that we don't have to have the
        # idx inside BasisFunctionInstance.
        phi_i_map: dict[int, BasisFunctionInstance] = {}

        # Exact pass: compute sphere distances from grid points of the blocks
        for s_idx, bx, by, bz in zip(*np.where(aabb_overlap_mask)):

            r = radius_i[s_idx]
            pos_v = pos_iv[s_idx]

            points_Rv = self.block_data.get_grid_points_xyz((bx, by, bz))
            dist2_R = np.sum((points_Rv - pos_v)**2, axis=-1)
            if not np.any(dist2_R <= r**2):
                continue

            # Overlap OK
            block_coords = (int(bx), int(by), int(bz))
            if block_coords not in block_map:
                block_map[block_coords] = GridBlock(
                    self.grid, block_coords,
                    self.block_data.block_start_c[bx, by, bz],
                    self.block_data.block_end_c[bx, by, bz])
            block = block_map[block_coords]

            sphere = spheres_i[s_idx]

            # Ugly, gotta create the phi instances here on the fly
            if s_idx not in phi_i_map.keys():
                new_phi = BasisFunctionInstance(
                    phi_idx,
                    sphere.phi_desc,
                    sphere.spline_idx,
                    sphere.parent_atom_idx,
                    sphere.first_mu,
                    sphere.pos_v,
                    sphere.cell_coords)
                phi_i_map[s_idx] = new_phi
                phi_idx += 1
                self.phi_i.append(new_phi)

            phi = phi_i_map[s_idx]
            block.phi_j.append(phi)

        # End sphere-block loop

        # Construct mapping for block-local phi_m => global phi_mu
        self.blocks = []
        for block in block_map.values():
            block.M_m = []
            block.phi_j = sorted(block.phi_j, key=lambda x: x.first_mu)
            block.phi_idx_j = []
            for phi in block.phi_j:
                mu = phi.first_mu
                for m in range(phi.get_num_mu()):
                    block.M_m.append(mu + m)

                block.phi_idx_j.append(phi.index)
            #
            self.blocks.append(block)
        #

    @dataclass
    class AtomUpdateResult:
        relpos_c: np.ndarray
        """Relative position of the atom after move"""
        rank: int
        """MPI rank of the atom now"""
        phi_spheres: list[PhiSphereData]
        """Metadata of phi spheres for this atom that now overlap the main
        cell. Includes periodic copies of phi in other cells."""

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
        if len(relpos_ac) != self.num_atoms:
            raise ValueError("Wrong atom position array shape")

        has_migrations = False
        old_relpos_ac = self.get_atom_positions(True)

        has_changes = False
        # Keep track of all phi that are now present
        phi_spheres = []

        # Find periodic copies ("image spheres") of all basis functions that
        # overlap the main cell. First a fast conservative pass that finds
        # potential images, then a slower exact check with blocked grid
        # that only keeps images that overlap with some grid point in this
        # MPI domain.

        for a in range(self.num_atoms):

            new_relpos = relpos_ac[a]
            if np.all(old_relpos_ac == new_relpos) and not force_update:
                # Nothing to do for this atom
                continue

            res = self._update_atom_position(a, new_relpos)

            has_changes = True
            assert np.all(res.relpos_c == new_relpos)

            if self._rank_a[a] != res.rank:
                has_migrations = True
                self._rank_a[a] = res.rank

            phi_spheres += res.phi_spheres

        # Have to rebuild internal phi lists if atoms moved around
        if has_changes or force_update:
            self._build_phi_instances(phi_spheres)

            if self.should_precalculate:
                self.precalculate_on_grid()

        self._relpos_ac = relpos_ac

        return has_migrations

    def _update_atom_position(
            self,
            atom_idx: int,
            new_relpos_c: np.ndarray) -> AtomUpdateResult:
        """"""
        new_rank = self.grid._gd.get_rank_from_position(new_relpos_c)

        """Handle periodic boundaries: basis funcs from other unit cells
        can extend to the "main" cell. Treat this as a problem of N
        spheres (phi) in a box, and we want to know which spheres overlap
        with a smaller box in the middle ("main" cell). We first have
        to construct the "periodic images" of spheres, then perform
        sphere-box overlap checks.
        """
        phi_spheres = []

        position_v = new_relpos_c @ self.grid.cell_cv

        spline_indices = self.spline_indices_for_a[atom_idx]
        mu = self._mu_range_a[atom_idx].start

        for i, phi in enumerate(self.get_phi_data_for_atom(atom_idx)):

            cells, positions_v = find_image_spheres_conservative(
                self.grid,
                position_v,
                phi.cutoff)

            spline_idx = spline_indices[i]

            for j, sphere_pos_v in enumerate(positions_v):
                phi_spheres.append(
                    PhiSphereData(sphere_pos_v, spline_idx, atom_idx, mu, phi,
                                  cell_coords=cells[j]))

            mu += (2 * phi.l + 1)

        return self.AtomUpdateResult(new_relpos_c, new_rank, phi_spheres)

    def _build_phi_instances(
            self,
            phi_spheres: list[PhiSphereData]) -> None:
        """"""

        # Input spheres are optimistic and may not actually overlap the cell.
        # Do blocking to cull them
        self._update_block_data(phi_spheres)
