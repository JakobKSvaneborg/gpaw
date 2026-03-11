import numpy as np
from gpaw.gpu import cupy as cp
from functools import cached_property
from dataclasses import dataclass, field
from gpaw.spline import Spline
from gpaw.core import UGDesc
from gpaw.old.grid_descriptor import GridBoundsError
from gpaw.new.timer import trace
from gpaw.setup import Setups, Setup
from typing import cast
from enum import Enum, auto
from abc import ABC, abstractmethod
import itertools

from typing import Protocol, runtime_checkable, TypeAlias


@runtime_checkable
class SplineLike(Protocol):
    """"""

    def __call__(self, x: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        ...


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


def sphere_overlaps_box(
    radius: float,
    center: np.ndarray,   # 3D array, real space pos of sphere center
    box_min: np.ndarray,  # 3D array, box "start" corner in real units
    box_max: np.ndarray,  # 3D array, box "end" corner in real units
) -> bool:
    """Exact sphere-box overlap check."""
    # closest point in the box to sphere center
    closest = np.maximum(box_min, np.minimum(center, box_max))
    dist2 = np.sum((closest - center)**2)
    return dist2 <= radius * radius


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


BlockCoords: TypeAlias = tuple[int, int, int]


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

    blocks_with_overlap: list[GridBlock] = field(default_factory=list)

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


class SplinePoolBase(ABC):
    """"""
    splines: list[SplineLike]

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

    def get_spline(self, spline_idx: int) -> SplineLike:
        """"""
        return self.splines[spline_idx]


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

    # Dict instead of list for more versatility if we modify this later:
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

    block_size: int

    blocks: list[GridBlock]
    # Store just those blocks that actually have phi overlap

    def __init__(self,
                 system: LFCSystemDesc,
                 use_gpu: bool = False,
                 mode: PrecalculationMode = PrecalculationMode.eIfPossible,
                 block_size: int | None = 8):
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
        block_size: int, optional
            Size of each grid block. None means no blocking.
        """
        self.grid = system.grid

        if block_size is not None:
            self.block_size = min(block_size, np.min(self.grid.mysize_c))
        else:
            self.block_size = np.max(self.grid.mysize_c)

        if len(system.atoms) == 0:
            raise RuntimeError("No atoms?!")

        self.xp = cp if use_gpu else np

        # TODO let caller choose dtype (32 or 64bit float)
        self.dtype = np.float64

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

        # This happens now inside block rebuild...
        # if self.should_precalculate:
        #    self.precalculate_on_grid()

    def uses_gpu(self) -> bool:
        """Returns True if the basis functions and splines are allocated in
        GPU memory."""
        return self.xp is not np

    def get_phi_data_for_atom(self, atom_idx: int) -> list[BasisFunctionDesc]:
        """"""
        out = []
        for spline_idx in self.spline_indices_for_a[atom_idx]:
            out.append(self.phi_datas[spline_idx])
        return out

    def get_spline(self, spline_idx: int) -> SplineLike:
        """"""
        return self.spline_pool.get_spline(spline_idx)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms in the system (not just in this MPI rank)."""
        return len(self.spline_indices_for_a.keys())

    def get_atom_positions(self, grid_relative=False) -> np.ndarray:
        """Returns current atom positions. Either position_av or relpos_ac"""
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

    @abstractmethod
    def _init_splines(self) -> None:
        """Creates and sets self.spline_pool"""
        raise NotImplementedError

    def _init_mu_lookups(self) -> None:
        """"""
        self._mu_range_a = []
        mu = 0
        for a in range(self.num_atoms):

            mu_local = 0
            for phi in self.get_phi_data_for_atom(a):
                mu_local += 2 * phi.l + 1

            self._mu_range_a.append(range(mu, mu + mu_local))
            mu += mu_local

    @trace
    def _update_grid_blocks(self) -> None:
        """Actually just fully rebuilds block info."""

        self.blocks = []
        grid_spacing_v = np.diag(self.grid._gd.h_cv)

        """For each phi, find which blocks they overlap. We start with an AABB
        (Axis Aligned Bounding Box) overlap test to find nearby boxes with
        possible overlap, then remove false positives by a proper spherical
        overlap test. The AABB check is done against boxes of size
        block_size^3, which can be too large if the grid domain doesn't divide
        evenly into blocks of this size. We compute real block shapes in the
        latter overlap test.
        """

        self._block_map: dict[BlockCoords, GridBlock] = {}

        # Can happen with domain decomposition, I guess? Nothing to block
        if self.grid.mysize_c.size == 0:
            return

        # Round division up. The last block is allowed to be smaller
        num_blocks_c = -(self.grid.mysize_c // -self.block_size).astype(int)
        real_block_size_v = self.block_size * grid_spacing_v

        for phi in self.get_phi_instances():
            # Compute AABB extents for the phi-sphere (real space units)
            radius = phi.get_cutoff()
            aabb_min = phi.position - radius
            aabb_max = phi.position + radius

            # Convert to be relative to origin of this grid domain
            rel_min_v = aabb_min - self.grid.start_c * grid_spacing_v
            rel_max_v = aabb_max - self.grid.start_c * grid_spacing_v

            # Compute block indices that the AABB overlaps
            min_block = np.floor(rel_min_v / real_block_size_v)
            max_block = np.floor(rel_max_v / real_block_size_v)

            min_block = np.maximum(min_block, 0).astype(int)
            max_block = np.minimum(max_block, num_blocks_c - 1).astype(int)

            for bx, by, bz in itertools.product(
                range(min_block[0], max_block[0] + 1),
                range(min_block[1], max_block[1] + 1),
                range(min_block[2], max_block[2] + 1),
            ):
                # Exact overlap test with real blocks: get block corners
                block_start_c = (self.grid.start_c
                                 + self.block_size * np.asarray([bx, by, bz]))

                # Clip end corner so that it stays within the MPI domain
                block_end_c = block_start_c + self.block_size
                block_end_c = np.minimum(block_end_c, self.grid.end_c)

                # FIXME can happen that a sphere DOES overlap the box,
                # but NOT any grid points in the box => not needed.
                # Happens already in LFCAtom.set_positions...
                if sphere_overlaps_box(
                    radius,
                    phi.position,
                    block_start_c * grid_spacing_v,
                    block_end_c * grid_spacing_v
                ):

                    block_coords = (bx, by, bz)
                    if block_coords in self._block_map:
                        block = self._block_map[block_coords]
                    else:
                        # first time in this block
                        block = GridBlock(
                            self.grid,
                            block_coords,
                            block_start_c,
                            block_end_c)
                        self._block_map[block_coords] = block

                    block.phi_j.append(phi)
                    phi.blocks_with_overlap.append(block)
                # end block loop

            # Sort the block list for convenience
            # phi.blocks_with_overlap = sorted(phi.blocks_with_overlap,
            #                                  key=lambda x : x.coords)

        # end phi loop

        # Construct mapping for block-local phi_m => global phi_mu
        for block in self._block_map.values():
            block.M_m = []
            block.phi_j = sorted(block.phi_j, key=lambda x: x.first_mu)
            block.phi_idx_j = []
            for phi in block.phi_j:
                mu = phi.first_mu
                for m in range(phi.get_num_mu()):
                    block.M_m.append(mu + m)

                block.phi_idx_j.append(phi.index)
            #
        #

    def get_relevant_blocks(self) -> list[GridBlock]:
        """Returns list of grid blocks that contribute, ie. have overlap with
        some phi."""
        return list(self._block_map.values())

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
    def has_precalculated_phi(self) -> bool:
        """"""
        raise NotImplementedError

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

    def is_gpu(self) -> bool:
        """True if this object is using GPU."""
        return self.xp is not np

    def num_basis_functions(self) -> int:
        """"""
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
        existing state."""
        relpos_ac = np.asanyarray(relpos_ac)
        if len(relpos_ac) != self.num_atoms:
            raise ValueError("Wrong atom position array shape")

        has_migrations = False
        old_relpos_ac = self.get_atom_positions(True)

        has_changes = False
        # Keep track of all phi that are now present
        phi_spheres = []

        for a in range(self.num_atoms):

            new_relpos = relpos_ac[a]
            if np.all(old_relpos_ac == new_relpos) and not force_update:
                # Nothing to do for this atom
                continue

            try:
                res = self._update_atom_position(a, new_relpos)
            except GridBoundsError as e:
                e.args = (f'Atom {a} too close to edge: {e}',)
                # FIXME if this happens our self._relpos_ac goes out of sync
                raise

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
        position_v = new_relpos_c @ self.grid.cell_cv
        L = np.diag(self.grid.cell_cv)  # shape = (3,)

        # Main cell "min" and "max" points. In principle just (0, 0, 0) and
        # (L, L, L), but in practice the last grid point can be one spacing
        # earlier, and that is the last point we need for overlap checks.
        # FIXME: (L, L, L) is safer because it also sees possible phi that
        # have radius smaller than the grid spacing. So go with L for now.
        # TODO: 1. Warn about small-radius phi
        #       2. Drop such phi from computations. More optimal and allows
        #          assertions checks like "did we only include phi that
        #          contribute"

        main_cell_start_v = np.asarray([0, 0, 0])
        main_cell_end_v = main_cell_start_v + L

        # Enforce no overlap in directions without periodic b.c.
        pbc_mask = ~np.array(self.grid.pbc_c)

        phi_spheres = []

        spline_indices = self.spline_indices_for_a[atom_idx]
        mu = self._mu_range_a[atom_idx].start

        for i, phi in enumerate(self.get_phi_data_for_atom(atom_idx)):

            spline_idx = spline_indices[i]

            # NOTE: there is a helper function in GridDescriptor called
            # get_boxes(), but it seems to give false positives. The following
            # tries to do better.
            r = phi.cutoff
            center = position_v

            # How many cells does the sphere overlap in given direction
            n_min = np.floor((center - r) / L).astype(int)
            n_max = np.floor((center + r) / L).astype(int)

            # TODO test if this actually works with non-pbc
            n_min = np.where(pbc_mask, 0, n_min)
            n_max = np.where(pbc_mask, 0, n_max)

            """Build a 3D shift grid. This Numpy magic is equivalent to:
            for nx, ny, nz in itertools.product(
                range(n_min[0], n_max[0] + 1),
                range(n_min[1], n_max[1] + 1),
                range(n_min[2], n_max[2] + 1),
            ):
                shift = np.array([nx, ny, nz])
                ...
            """
            ranges = [np.arange(n_min[d], n_max[d] + 1) for d in range(3)]
            grids = np.meshgrid(*ranges, indexing="ij")
            shifts = np.stack(grids, axis=-1).reshape(-1, 3)
            # includes (0, 0, 0), ie. the main cell
            assert shifts.size > 0

            # Where the periodic copy sphere would be
            translated_centers = center - shifts * L

            # Which shifted sphere actually overlap with the (0,0,0) cell.

            mask = np.asarray(
                [sphere_overlaps_box(r, sphere_pos,
                                     main_cell_start_v,
                                     main_cell_end_v)
                 for sphere_pos in translated_centers]
            )
            assert np.any(mask)

            for atom_pos in translated_centers[mask]:
                phi_spheres.append(
                    PhiSphereData(atom_pos, spline_idx, atom_idx, mu, phi))

            mu += (2 * phi.l + 1)

        # FIXME: the above can still give phi instances that technically
        # overlap the cell, but don't overlap with any grid point...?
        # Not a big issue but wastes memory/flops
        return self.AtomUpdateResult(new_relpos_c, new_rank, phi_spheres)

    def _build_phi_instances(
            self,
            phi_spheres: list[PhiSphereData]) -> None:
        """"""
        self.phi_i = []

        i = 0
        for sphere in phi_spheres:
            new_phi = BasisFunctionInstance(
                i,
                sphere.phi_desc,
                sphere.spline_idx,
                sphere.parent_atom_idx,
                sphere.first_mu,
                sphere.pos_v)

            self.phi_i.append(new_phi)

        self._update_grid_blocks()

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

    def calculate_potential_matrices(
            self,
            vt_G: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """Alias for calculate_potential_matrix"""
        V_MM = self.calculate_potential_matrix(vt_G)
        return V_MM[np.newaxis]

    def get_cache_size_estimate(self) -> int:
        """Estimate of how many bytes are needed to precalculate and cache the
        basis functions on the grid.
        """
        num_sites = 0
        for block in self.get_relevant_blocks():
            num_sites += int(np.prod(block.shape))

        dummy = np.empty((1), dtype=self.dtype)
        return num_sites * self.num_basis_functions() * dummy.itemsize
