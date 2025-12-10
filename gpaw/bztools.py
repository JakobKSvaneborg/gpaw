from itertools import product

import numpy as np
from ase.dft.kpoints import monkhorst_pack
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from ase import Atoms

try:
    from scipy.spatial import QhullError
except ImportError:  # scipy < 1.8
    from scipy.spatial.qhull import QhullError

from gpaw import GPAW
from gpaw.mpi import normalize_communicator
from gpaw.old.kpt_descriptor import kpts2sizeandoffsets, to1bz
from gpaw.symmetry import Symmetry, aglomerate_points


def get_lattice_symmetry(cell_cv, tolerance=1e-7):
    """Return symmetry object of lattice group.

    Parameters
    ----------
    cell_cv : ndarray
        Unit cell.

    Returns
    -------
    gpaw.symmetry object

    """
    # NB: Symmetry.find_lattice_symmetry() uses self.pbc_c, which defaults
    # to pbc along all three dimensions. Hence, it seems that the lattice
    # symmetry transformations produced by this method could be faulty if
    # there are non-periodic dimensions in a system. XXX
    latsym = Symmetry([0], cell_cv, tolerance=tolerance)
    latsym.find_lattice_symmetry()
    return latsym


def find_high_symmetry_monkhorst_pack(atoms: Atoms,
                                      density: float,
                                      world=None):
    """Make high symmetry Monkhorst Pack k-point grid.

    Searches for and returns a Monkhorst Pack grid which
    contains the corners of the irreducible BZ so that when the
    number of k-points are reduced the full irreducible brillouion
    zone is spanned.

    Parameters
    ----------
    atoms : Atoms
        Atoms object to find symmetries of.
    density : float
        The required minimum density of the Monkhorst Pack grid.

    Returns
    -------
    ndarray
        Array of shape (nk, 3) containing the k-points.

    """
    world = normalize_communicator(world)

    if not isinstance(atoms, Atoms):
        raise TypeError(f'Use atoms instead of {type(atoms)}.')

    pbc = atoms.pbc
    minsize, offset = kpts2sizeandoffsets(density=density, even=True,
                                          gamma=True, atoms=atoms)

    # NB: get_bz() and get_bz_from_atoms() wants a pbc_c, but never gets it.
    # The pbc will therefore fall back to True along all dimensions.
    # NB: Why return latibzk_kc, if we never use it? XXX
    bzk_kc, ibzk_kc, latibzk_kc = get_bz_from_atoms(atoms)

    maxsize = minsize + 9
    minsize[~pbc] = 1
    maxsize[~pbc] = 2

    if world.rank == 0:
        print('Brute force search for symmetry ' +
              'complying MP-grid... please wait.')

    for n1 in range(minsize[0], maxsize[0], 2):
        for n2 in range(minsize[1], maxsize[1], 2):
            for n3 in range(minsize[2], maxsize[2], 2):
                size = n1, n2, n3
                size, offset = kpts2sizeandoffsets(size=size, gamma=True,
                                                   atoms=atoms)

                ints = ((ibzk_kc + 0.5 - offset) * size - 0.5)[:, pbc]

                if (np.abs(ints - np.round(ints)) < 1e-5).all():
                    kpts_kc = monkhorst_pack(size) + offset
                    kpts_kc = to1bz(kpts_kc, atoms.cell)

                    for ibzk_c in ibzk_kc:
                        diff_kc = np.abs(kpts_kc - ibzk_c)[:, pbc].round(6)
                        if not (np.mod(np.mod(diff_kc, 1), 1) <
                                1e-5).all(axis=1).any():
                            raise AssertionError('Did not find ' + str(ibzk_c))
                    if world.rank == 0:
                        print('Done. Monkhorst-Pack grid:', size, offset)
                    return {'size': size, 'gamma': True}

    if world.rank == 0:
        print(ibzk_kc.round(5))

    raise RuntimeError('Did not find matching k-points for the IBZ')


def unfold_points(points, U_scc, tol=1e-8, mod=None):
    """Unfold k-points using a given set of symmetry operators.

    Parameters
    ----------
    points: ndarray
    U_scc: ndarray
    tol: float
        Tolerance indicating when k-points are considered to be
        identical.
    mod: integer 1 or None
        Consider k-points spaced by a full reciprocal lattice vector
        to be identical.

    Returns
    -------
    ndarray
        Array of shape (nk, 3) containing the unfolded k-points.
    """

    points = np.concatenate(np.dot(points, U_scc.transpose(0, 2, 1)))
    return unique_rows(points, tol=tol, mod=mod)


def unique_rows(ain, tol=1e-10, mod=None, aglomerate=True):
    """Return unique rows of a 2D ndarray.

    Parameters
    ----------
    ain : 2D ndarray
    tol : float
        Tolerance indicating when k-points are considered to be
        identical.
    mod : integer 1 or None
        Consider k-points spaced by a full reciprocal lattice vector
        to be identical.
    aglomerate : bool
        Aglomerate clusters of points before comparing.

    Returns
    -------
    2D ndarray
        Array containing only unique rows.
    """
    # Move to positive octant
    a = ain - ain.min(0)

    # First take modulus
    if mod is not None:
        a = np.mod(np.mod(a, mod), mod)

    # Round and take modulus again
    if aglomerate:
        aglomerate_points(a, tol)
    a = a.round(-np.log10(tol).astype(int))
    if mod is not None:
        a = np.mod(a, mod)

    # Now perform ordering
    order = np.lexsort(a.T)
    a = a[order]

    # Find unique rows
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(1)

    return ain[order][ui]


def get_smallest_Gvecs(cell_cv, n=5):
    """Find smallest reciprocal lattice vectors.

    Parameters
    ----------
    cell_cv : ndarray
        Unit cell.
    n : int
        Sampling along each crystal axis.

    Returns
    -------
    G_xv : ndarray
        Reciprocal lattice vectors in cartesian coordinates.
    N_xc : ndarray
        Reciprocal lattice vectors in crystal coordinates.

    """
    B_cv = 2.0 * np.pi * np.linalg.inv(cell_cv).T
    N_xc = np.indices((n, n, n)).reshape((3, n**3)).T - n // 2
    G_xv = N_xc @ B_cv

    return G_xv, N_xc


def get_symmetry_operations(U_scc, time_reversal):
    """Return point symmetry operations."""

    if U_scc is None:
        U_scc = np.array([np.eye(3)])

    inv_cc = -np.eye(3, dtype=int)
    has_inversion = (U_scc == inv_cc).all(2).all(1).any()

    if has_inversion:
        time_reversal = False

    if time_reversal:
        Utmp_scc = np.concatenate([U_scc, -U_scc])
    else:
        Utmp_scc = U_scc

    return Utmp_scc


def get_ibz_vertices(cell_cv, U_scc=None, time_reversal=None,
                     origin_c=None):
    """Determine irreducible BZ.

    Parameters
    ----------
    cell_cv : ndarray
        Unit cell
    U_scc : ndarray
        Crystal symmetry operations.
    time_reversal : bool
        Use time reversal symmetry?

    Returns
    -------
    ibzk_kc : ndarray
        Vertices of the irreducible BZ.
    """
    # Choose an origin
    if origin_c is None:
        origin_c = np.array([0.12, 0.22, 0.21], float)
    else:
        assert (np.abs(origin_c) < 0.5).all()

    if U_scc is None:
        U_scc = np.array([np.eye(3)])

    if time_reversal is None:
        time_reversal = False

    Utmp_scc = get_symmetry_operations(U_scc, time_reversal)

    icell_cv = np.linalg.inv(cell_cv).T
    B_cv = icell_cv * 2 * np.pi
    A_cv = np.linalg.inv(B_cv).T

    # Map a random point around
    point_sc = np.dot(origin_c, Utmp_scc.transpose((0, 2, 1)))
    assert len(point_sc) == len(unique_rows(point_sc))
    point_sv = np.dot(point_sc, B_cv)

    # Translate the points
    n = 5
    G_xv, N_xc = get_smallest_Gvecs(cell_cv, n=n)
    G_xv = np.delete(G_xv, n**3 // 2, axis=0)

    # Mirror points in plane
    N_xv = G_xv / (((G_xv**2).sum(1))**0.5)[:, np.newaxis]

    tp_sxv = (point_sv[:, np.newaxis] - G_xv[np.newaxis] / 2.)
    delta_sxv = ((tp_sxv * N_xv[np.newaxis]).sum(2)[..., np.newaxis] *
                 N_xv[np.newaxis])
    points_xv = (point_sv[:, np.newaxis] - 2 * delta_sxv).reshape((-1, 3))
    points_xv = np.concatenate([point_sv, points_xv])
    try:
        voronoi = Voronoi(points_xv)
    except QhullError:
        return get_ibz_vertices(cell_cv, U_scc=U_scc,
                                time_reversal=time_reversal,
                                origin_c=origin_c + [0.01, -0.02, -0.01])

    ibzregions = voronoi.point_region[0:len(point_sv)]

    ibzregion = ibzregions[0]
    ibzk_kv = voronoi.vertices[voronoi.regions[ibzregion]]
    ibzk_kc = np.dot(ibzk_kv, A_cv.T)

    return ibzk_kc


def get_bz(calc, pbc_c=np.ones(3, bool)):
    """Return the BZ and IBZ vertices.

    Parameters
    ----------
    calc : str, GPAW calc instance

    Returns
    -------
    bzk_kc : ndarray
        Vertices of BZ in crystal coordinates
    ibzk_kc : ndarray
        Vertices of IBZ in crystal coordinates

    """

    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)
    cell_cv = calc.wfs.gd.cell_cv

    # Crystal symmetries
    symmetry = calc.wfs.kd.symmetry
    cU_scc = get_symmetry_operations(symmetry.op_scc,
                                     symmetry.time_reversal)

    return get_reduced_bz(cell_cv, cU_scc, False, pbc_c=pbc_c)


def get_bz_from_atoms(atoms):
    pbc_c = atoms.pbc
    cell_cv = atoms.cell
    from gpaw.symmetry import Symmetry
    id_a = atoms.get_chemical_symbols()
    symmetry = Symmetry(id_a, atoms.cell, atoms.pbc)
    symmetry.analyze(atoms.get_scaled_positions())
    cU_scc = get_symmetry_operations(symmetry.op_scc,
                                     symmetry.time_reversal)

    return get_reduced_bz(cell_cv, cU_scc, False, pbc_c=pbc_c)


def get_reduced_bz(cell_cv, cU_scc, time_reversal,
                   pbc_c=np.ones(3, bool), tolerance=1e-7):

    """Reduce the BZ using the crystal symmetries to obtain the IBZ.

    Parameters
    ----------
    cell_cv : ndarray
        Unit cell.
    cU_scc : ndarray
        Crystal symmetry operations.
    time_reversal : bool
        Switch for time reversal.
    pbc: bool or [bool, bool, bool]
        Periodic bcs
    """

    if time_reversal:
        # NB: The method never seems to be called with time_reversal=True,
        # and hopefully get_bz() will generate the right symmetry operations
        # always. So, can we remove this input? XXX
        cU_scc = get_symmetry_operations(cU_scc, time_reversal)

    # Lattice symmetries
    latsym = get_lattice_symmetry(cell_cv, tolerance=tolerance)
    lU_scc = get_symmetry_operations(latsym.op_scc,
                                     latsym.time_reversal)

    # Find Lattice IBZ
    ibzk_kc = get_ibz_vertices(cell_cv,
                               U_scc=latsym.op_scc,
                               time_reversal=latsym.time_reversal)
    latibzk_kc = ibzk_kc.copy()

    # Expand lattice IBZ to crystal IBZ
    ibzk_kc = expand_ibz(lU_scc, cU_scc, ibzk_kc, pbc_c=pbc_c)

    # Fold out to full BZ
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               cU_scc.transpose(0, 2, 1))))

    return bzk_kc, ibzk_kc, latibzk_kc


def expand_ibz(lU_scc, cU_scc, ibzk_kc, pbc_c=np.ones(3, bool), world=None):
    """Expand IBZ from lattice group to crystal group.

    Parameters
    ----------
    lU_scc : ndarray
        Lattice symmetry operators.
    cU_scc : ndarray
        Crystal symmetry operators.
    ibzk_kc : ndarray
        Vertices of lattice IBZ.

    Returns
    -------
    ibzk_kc : ndarray
        Vertices of crystal IBZ.

    """

    # Find right cosets. The lattice group is partioned into right cosets of
    # the crystal group. This can in practice be done by testing whether
    # U1 U2^{-1} is in the crystal group as done below.
    world = normalize_communicator(world)

    cosets = []
    Utmp_scc = lU_scc.copy()
    while len(Utmp_scc):
        U1_cc = Utmp_scc[0].copy()
        Utmp_scc = np.delete(Utmp_scc, 0, axis=0)
        j = 0
        new_coset = [U1_cc]
        while j < len(Utmp_scc):
            U2_cc = Utmp_scc[j]
            U3_cc = np.dot(U1_cc, np.linalg.inv(U2_cc))
            if (U3_cc == cU_scc).all(2).all(1).any():
                new_coset.append(U2_cc)
                Utmp_scc = np.delete(Utmp_scc, j, axis=0)
                j -= 1
            j += 1
        cosets.append(new_coset)

    volume = np.inf
    nibzk_kc = ibzk_kc
    U0_cc = cosets[0][0]  # Origin

    if np.any(~pbc_c):
        nonpbcind = np.argwhere(~pbc_c)

    # Once the coests are known the irreducible zone is given by picking one
    # operation from each coset. To make sure that the IBZ produced is simply
    # connected we compute the volume of the convex hull of the produced IBZ
    # and pick (one of) the ones that have the smallest volume. This is done by
    # brute force and can sometimes take a while, however, in most cases this
    # is not a problem.
    combs = list(product(*cosets[1:]))[world.rank::world.size]
    for U_scc in combs:
        if not len(U_scc):
            continue
        U_scc = np.concatenate([np.array(U_scc), [U0_cc]])
        tmpk_kc = unfold_points(ibzk_kc, U_scc)
        volumenew = convex_hull_volume(tmpk_kc)

        if np.any(~pbc_c):
            # Compute the area instead
            volumenew /= (tmpk_kc[:, nonpbcind].max() -
                          tmpk_kc[:, nonpbcind].min())

        if volumenew < volume:
            nibzk_kc = tmpk_kc
            volume = volumenew

    ibzk_kc = unique_rows(nibzk_kc)
    volume = np.array((volume,))

    volumes = np.zeros(world.size, float)
    world.all_gather(volume, volumes)

    minrank = np.argmin(volumes)
    minshape = np.array(ibzk_kc.shape)
    world.broadcast(minshape, minrank)

    if world.rank != minrank:
        ibzk_kc = np.zeros(minshape, float)
    world.broadcast(ibzk_kc, minrank)

    return ibzk_kc


def tetrahedron_volume(a, b, c, d):
    """Calculate volume of tetrahedron.

    Parameters
    ----------
    a, b, c, d : ndarray
        Vertices of tetrahedron.

    Returns
    -------
    float
        Volume of tetrahedron.

    """
    return np.abs(np.einsum('ij,ij->i', a - d,
                            np.cross(b - d, c - d))) / 6


def convex_hull_volume(pts):
    """Calculate volume of the convex hull of a collection of points.

    Parameters
    ----------
    pts : list, ndarray
        A list of 3d points.

    Returns
    -------
    float
        Volume of convex hull.

    """
    hull = ConvexHull(pts)
    dt = Delaunay(pts[hull.vertices])
    tets = dt.points[dt.simplices]
    vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                    tets[:, 2], tets[:, 3]))
    return vol


def mp_from_density(
        atoms: Atoms, density: float, *, 
        symmetry: bool,
        gamma: bool,
        even: bool,
        high_symmetry_points: bool,
        symmetric_bz: bool, 
        comm
) -> tuple[np.ndarray, int]:
    """
    Get a monkhorstpack grid with the lowest number of k-points in the
    reducible/irreducible Brillouin zone that still satisfies a given
    minimum distance condition in the real space (nx, ny, nz)-supercell.

    :param atoms: Atoms object used to the k-points
    :param density: Density of k-points (kpts/Å^-1)
    :param symmetry: Consider symmetry (irreducible) or not (reducible)
        when determining the k-point grid.
    :param comm: Communicator used to parallelize the symmetry reduction.
    :return: np.array([nx, ny, nz]), number of points in the brillouin
        zone
    """
    # For orthogonal cells: min_distance = 2 * np.pi * density
    min_distance = 2 * np.pi * density
    return mindistance2monkhorstpack(
        atoms=atoms,
        min_distance=min_distance,
        maxperdim='auto',
        symmetry=symmetry,
        even=even,
        gamma=gamma,
        high_symmetry_points=high_symmetry_points,
        symmetric_bz=symmetric_bz,
        comm=comm,
    )

def mindistance2monkhorstpack(
    atoms, *, min_distance, 
    maxperdim=16, 
    even, 
    symmetry,
    gamma,
    high_symmetry_points,
    symmetric_bz,
    comm
) -> tuple[np.ndarray, int]:
    """
    If symmetry==False (default), find a Monkhorst-Pack grid
    (nx, ny, nz) with lowest number of k-points in the *reducible*
    Brillouin zone, which still satisfying a given minimum distance
    (`min_distance`) condition in real space (nx, ny, nz)-supercell.
    Returns kpt_c.

    If symmetry==True (requires gpaw), returns the lowest number
    of k-points in the *irreducible* Brillouin zone, with same
    minimum distance condition.
    Returns a tuple (kpt_c, nibz).

    Compared to ase.calculators.calculator kptdensity2monkhorstpack
    routine, this metric is based on a physical quantity (real space
    distance), and it doesn't depend on non-physical quantities, such as
    the cell vectors, since basis vectors can be always transformed
    with integer determinant one matrices. In other words, it is
    invariant to particular choice of cell representations.

    On orthogonal cells, min_distance = 2 * np.pi * kptdensity.
    """
    if symmetry:
        # XXX Needs replacement by some new GPAW object
        from gpaw.old.kpt_descriptor import KPointDescriptor
        from gpaw.symmetry import Symmetry

        id_a = atoms.get_chemical_symbols()
        symmetry = Symmetry(id_a, atoms.cell, atoms.pbc)
        symmetry.analyze(atoms.get_scaled_positions())

        def get_nibz(nkpts_c):
            # Note: Neglects magnetic moments for now
            kpts_kc = monkhorst_pack(nkpts_c)
            kpts_kc -= 1/(2*nkpts_c)
            kd = KPointDescriptor(kpts_kc)
            print(kpts_kc)
            kd.set_symmetry(atoms, symmetry)
            if -1 in kd.bz2bz_ks:
                # Disfavor unsymmetric
                return 100000
            return len(kd.ibzk_kc)

        key = get_nibz
    else:

        def get_nk(nkpts_c):
            return np.prod(nkpts_c)

        key = get_nk  # lambda nkpts_c: np.prod(nkpts_c)

    _maxperdim = maxperdim if maxperdim != 'auto' else 16
    while 1:
        try:
            kpt_c = _mindistance2monkhorstpack(
                atoms.cell,
                atoms.pbc,
                min_distance,
                _maxperdim,
                even,
                key,
                comm=comm,
            )
        except KPTGridNotFound:
            if maxperdim != 'auto':
                raise
            print(
                f'kpt grid not found with maxperdim={_maxperdim}, doubling it.'
            )
            _maxperdim *= 2
            continue
        break

    return kpt_c, key(kpt_c)


def _mindistance2monkhorstpack(
    cell,
    pbc_c,
    min_distance,
    maxperdim,
    even,
    key=lambda nkpts_c: np.prod(nkpts_c),
    *,
    comm,
):
    from ase import Atoms
    from ase.neighborlist import NeighborList

    step = 2 if even else 1
    nl = NeighborList(
        [min_distance / 2], skin=0.0, self_interaction=False, bothways=False
    )

    def err():
        raise KPTGridNotFound(
            'Could not find a proper k-point grid for the '
            'system. Try running with a larger maxperdim.'
        )

    def check(nkpts_c):
         nl.update(Atoms('H', cell=cell @ np.diag(nkpts_c), pbc=pbc_c))
        return len(nl.get_neighbors(0)[1]) == 0

    rank, size = (0, 1)
    # rank, size = (comm.rank, comm.size)
    ranges = [
        range(step, maxperdim + 1, step) if pbc else range(1, 2)
        for pbc in pbc_c
    ]
    kpts_nc = np.column_stack([*map(np.ravel, np.meshgrid(*ranges))])[
        rank::size
    ]
    kpts_nx = np.array(
        [[*nkpts_c, key(nkpts_c)] for nkpts_c in kpts_nc if check(nkpts_c)]
    )
    if len(kpts_nx):
        minid = np.argmin(kpts_nx[:, 3])
        minkpt_x = kpts_nx[minid]
    else:
        err()
        # To enable parallelization, this is required
        minkpt_x = np.array([0, 0, 0, 100_000_000], dtype=int)
    # if comm is None:
    if 1:
        # XXX DISABLED PARALLEL CODE DUE TO APPARENT BUG.  --askhl
        len(minkpt_x) or err()
        value_c = minkpt_x[:3]
        return value_c

    minkpt_rx = np.zeros((4 * comm.size,), dtype=int)
    comm.all_gather(minkpt_x, minkpt_rx)
    minkpt_rx = minkpt_rx.reshape((-1, 4))
    value_c = minkpt_rx[np.argmin(minkpt_rx[:, 3]), :3]
    value_c[0] > 0 or err()
    print('VALUE_C', value_c.shape)
    assert len(value_c) == 3, 'BROKEN, or so I think.'
    return value_c



if __name__ == "__main__":
    def kpoint_generator(even=True):
        assert even
        for k1 in range(2, 18):
            for k2 in range(2, 18):
                for k3 in range(2, 18):
                    yield np.array((k1,k2,k3))

    def is_even(kpt_c):
        return np.allclose(kpt_c % 2, [0,0,0])

    print(filter(is_even, kpoint_generator()))

# XXX
# Demo code starts here
import numpy as np
from ase.neighborlist import NeighborList
from ase import Atoms
from ase.dft.kpoints import monkhorst_pack


class KPoint:
    def __init__(self, kpt_c):
        self.kpt_c = np.array(kpt_c)

    def is_even(self):
        return np.allclose(self.kpt_c % 2, [0,0,0])

    def __repr__(self):
        return str(self.kpt_c)

def kpoint_generator(maxk=10):
    for k1 in range(maxk +1, 2, -1):
        for k2 in range(2, maxk+1):
            for k3 in range(2, maxk+1):
                yield KPoint((k1,k2,k3))

def is_even(kpoint: KPoint):
    return kpoint.is_even()

def min_distance(distance, atoms):
    nl = NeighborList(
        [distance / 2], skin=0.0, self_interaction=False, bothways=False
    )
    def _min_distance(kpoint):
        nl.update(Atoms('H', cell=atoms.cell @ np.diag(kpoint.kpt_c), pbc=atoms.pbc))
        return len(nl.get_neighbors(0)[1]) == 0

    return _min_distance

from gpaw.old.kpt_descriptor import KPointDescriptor
from gpaw.symmetry import Symmetry

def symmetric_mp_grid(atoms):
    def _symmetric_mp_grid(kpoint):
        id_a = atoms.get_chemical_symbols()
        symmetry = Symmetry(id_a, atoms.cell, atoms.pbc)
        symmetry.analyze(atoms.get_scaled_positions())
        nkpts_c = kpoint.kpt_c
        kpts_kc = monkhorst_pack(nkpts_c)
        kpts_kc -= 1/(2*nkpts_c)
        kd = KPointDescriptor(kpts_kc)
        kd.set_symmetry(atoms, symmetry)
        if -1 in kd.bz2bz_ks:
            # Disfavor unsymmetric
            return False

        kpoint.nibz = len(kd.ibzk_kc)
        return True

    return _symmetric_mp_grid



from ase.build import bulk
#atoms = bulk('Si')
from ase.io import read
atoms = read('/home/kuisma/a.xyz')
predicates = [is_even, min_distance(12, atoms), symmetric_mp_grid(atoms)]

def satisfies_all(x):
    for pred in predicates:
        if not pred(x):
            return False
    return True

lst = []
for kpoint in filter(satisfies_all, kpoint_generator()):
    print(kpoint)
    lst.append(kpoint)

def sort(kpoint):
    return kpoint.nibz

print('sorted:')
for kpt in sorted(lst, key=sort):
    print(kpt)


