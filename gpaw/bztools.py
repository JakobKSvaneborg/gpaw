from itertools import product
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.dft.kpoints import monkhorst_pack
from ase.neighborlist import NeighborList
from scipy.spatial import ConvexHull, Delaunay, Voronoi

try:
    from scipy.spatial import QhullError
except ImportError:  # scipy < 1.8
    from scipy.spatial.qhull import QhullError

from gpaw import GPAW
from gpaw.mpi import SerialCommunicator
from gpaw.new import zips
from gpaw.new.brillouin import BZPoints
from gpaw.new.symmetry import create_symmetries_object
from gpaw.old.kpt_descriptor import to1bz
from gpaw.symmetry import Symmetry, aglomerate_points
from gpaw.utilities.symmetry import find_lattice_symmetry


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


def optimal_monkhorst_pack_grid(
        atoms: Atoms,
        *,
        kptdensity: float,
        minimize_ibz_points: bool = False,
        force_even: bool = False,
        force_gamma: bool = False,
        contains_ibz_vertices: bool = False,
        is_symmetric_mp_grid: bool = False,
        nmaxperdim: int = 8) -> dict[str, Any]:
    """
    This function returns a dictionary yielding a Monkhorst-Pack k-point
    sampling which is compliant with all desired conditions.

    Parameters
    ----------
    atoms:
        Atoms object to generate the Monkhorst-Pack grid for.
    kptdensity:
        The required minimum density of the k-points (kpts/Å^-1). This
        density is related to the minimum real-space distance between
        equivalent points in the Born-von Kármán supercell by
        min_distance = 2𝜋 * kptdensity.
    minimize_ibz_points:
        Should this function return the compliant MP grid which contains the
        smallest number of IBZ points (True) or the MP grid which contains the
        smallest number of BZ points (False).
    force_even:
        Should the Monkhorst-Pack grids be forced to contain an even number of
        k-points along the crystal axes?
    force_gamma:
        Should the k-point sampling be forced to contain the Gamma point
        through offsets?
    contains_ibz_vertices:
        Should the function only accept k-point samplings which contain all
        vertices of the IBZ?
    is_symmetric_mp_grid:
        Should the function only accept k-point samplings which are as
        symmetric as the crystal?
    nmaxperdim:
        When searching for compliant k-point samplings, how large should the
        maximum sampling size be (along each crystal axes) compared to the
        minimum size.

    Returns
    -------
        Dictionary yielding a Monkhorst-Pack k-point sampling compliant with
        all desired predicates.

    """

    msg = ('Searching for a Monkhorst-Pack grid which is '
           'compliant with the following predicates:')

    msg += ('\n\nThe grid should effectively yield a Born-von Kármán supercell'
            f'that does not repeat for at least 2𝜋 * {kptdensity} Å.')

    predicate_functions = []
    if contains_ibz_vertices:
        predicate_functions.append(contains_ibz_vertices_predicate)
        msg += ('\n\nThe k-point sampling must contain every vertex point '
                'of the irreducible Brillouin zone.')
    if is_symmetric_mp_grid:
        predicate_functions.append(is_symmetric_mp_grid_predicate)
        msg += '\n\nThe k-point sampling must be as symmetric as the crystal.'

    print(msg)

    minsize = get_mp_grid_from_min_distance_criteria(
        atoms, 2. * np.pi * kptdensity, even=force_even)

    if force_even:
        minsize += minsize % 2
        step = 2
    else:
        step = 1
    maxsize = minsize + nmaxperdim

    pbc = atoms.pbc
    minsize[~pbc] = 1
    maxsize[~pbc] = 1

    def mp_gridsize_generator(minsize, maxsize, step):
        for size in product(
                range(minsize[0], maxsize[0] + 1, step),
                range(minsize[1], maxsize[1] + 1, step),
                range(minsize[2], maxsize[2] + 1, step)):
            yield size
    mp_grids = np.array(list(mp_gridsize_generator(minsize, maxsize, step)))

    for predicate_function in predicate_functions:
        mp_grids = mp_grids[predicate_function(mp_grids, atoms,
                                               gamma=force_gamma)]
        if len(mp_grids) == 0:
            raise RuntimeError('Could not find grid which satisfies the'
                               f' {predicate_function.__name__}')

    if len(mp_grids) == 1:
        return {'size': mp_grids[0], 'gamma': force_gamma}
    else:
        if minimize_ibz_points:
            nk_ibz = get_nk_ibz(mp_grids, atoms, force_gamma)
            return {'size': mp_grids[np.argmin(nk_ibz)],
                    'gamma': force_gamma}
        else:
            nk_bz = np.prod(mp_grids, axis=1)
            return {'size': mp_grids[np.argmin(nk_bz)],
                    'gamma': force_gamma}


def get_mp_grid_from_min_distance_criteria(atoms, min_distance, even):
    """
    Get a Monkhorst-Pack grid with the lowest number of k-points in the
    Brillouin zone that still satisfies a given minimum distance condition in
    the real-space Born-von Kármán supercell.

    Compared to the method kptdensity2monkhorstpack from
    ase.calculators.calculator, this metric is based on a physical quantity
    (real-space distance), and it does not depend on non-physical quantities
    such as the choice of cell vectors which can be always be transformed by
    determinant=±1 matrices (isometries). In other words, this method is
    invariant to the choice of cell representation.

    For orthogonal cells, min_distance = 2𝜋 * kptdensity.
    """

    minsize_naive = kptdensity2monkhorstpack(
        atoms, kptdensity=min_distance / (2. * np.pi), even=even)
    minsize = -(minsize_naive // -2)  # ceiling division :)

    if even:
        minsize_naive += minsize_naive % 2
        minsize += minsize % 2
        step = 2
    else:
        step = 1

    pbc_c = atoms.pbc
    cell_cv = atoms.cell
    minsize[~pbc_c] = 1
    minsize_naive[~pbc_c] = 1

    compliant_mp_grids = []
    nk = []
    for size in product(
            range(minsize[0], minsize_naive[0] + 1, step),
            range(minsize[1], minsize_naive[1] + 1, step),
            range(minsize[2], minsize_naive[2] + 1, step)):

        neighborlist = NeighborList([min_distance / 2], skin=0.0,
                                    self_interaction=False, bothways=False)
        neighborlist.update(
            Atoms('H', cell=np.diag(size) @ cell_cv, pbc=pbc_c))

        if len(neighborlist.get_neighbors(0)[1]) == 0:
            compliant_mp_grids.append(size)
            nk.append(np.prod(size))

    if len(compliant_mp_grids) == 0:
        # This should never happen since at least
        # minsize_naive should comply with the min_distance criteria.
        # Remove this check?
        raise RuntimeError('Did not find compliant k-points grid'
                           'with minimum real-space distance criteria.')

    best_grid = compliant_mp_grids[np.argmin(nk)]
    return np.array(best_grid)


def contains_ibz_vertices_predicate(mp_grids,
                                    atoms: Atoms,
                                    gamma: bool):
    """For a list of Monkhorst-Pack grid sizes, this function checks whether
    each k-point sampling contains the vertices of the irreducible
    Brillouin zone.

    Parameters
    ----------
    mp_grids:
        List of Monkhorst-Pack grid sizes.
    atoms:
        Atoms object which is used to generate a list of symmetries.
    gamma:
        Boolean for whether the k-point sampling should be forced to contain
        the Gamma point through offsets. This should always be True for this
        function and is here as a safety check.

    Returns
    -------
        An array with a boolean for each grid size representing whether the
        predicate is satisfied by the k-point sampling.

    """

    pbc_c = atoms.pbc
    cell_cv = atoms.cell

    if not gamma:
        raise ValueError('You cannot get an MP grid that contains all '
                         'IBZ vertices without first forcing the '
                         'inclusion of the Gamma-point.')
    if not (mp_grids[:, pbc_c] % 2 == 0).all():
        raise ValueError('You cannot get an MP grid that contains all '
                         'IBZ vertices without an even k-point sampling.')

    # Get IBZ vertices in lattice group.
    lU_scc = find_lattice_symmetry(cell_cv, pbc_c, tol=1e-5)
    latibz_vert_kc = get_ibz_vertices(cell_cv, U_scc=lU_scc,
                                      time_reversal=False)

    # Expand IBZ vertices from lattice group to crystal group.
    cU_scc = create_symmetries_object(atoms).rotation_scc
    ibzk_kc = expand_ibz(lU_scc, cU_scc, latibz_vert_kc,
                         comm=SerialCommunicator(), pbc_c=pbc_c)

    bools = np.zeros(len(mp_grids), dtype=bool)
    for count, size in enumerate(np.array(mp_grids)):
        offsets = 0.5 / size
        offsets[~pbc_c] = 0

        ints = ((ibzk_kc + 0.5 - offsets) * size - 0.5)[:, pbc_c]

        if (np.abs(ints - np.round(ints)) < 1e-5).all():
            kpts_kc = monkhorst_pack(size) + offsets
            kpts_kc = to1bz(kpts_kc, cell_cv)

            for ibzk_c in ibzk_kc:
                diff_kc = np.abs(kpts_kc - ibzk_c)[:, pbc_c].round(6)
                if not ((diff_kc % 1) % 1 < 1e-5).all(axis=1).any():
                    raise AssertionError('Did not find vertex ' + str(ibzk_c))
            bools[count] = True

    return bools


def is_symmetric_mp_grid_predicate(mp_grids,
                                   atoms: Atoms,
                                   gamma: bool):
    """For a list of Monkhorst-Pack grid sizes, this function checks whether
    each k-point sampling is as symmetric as the crystal.

    Parameters
    ----------
    mp_grids
        List of Monkhorst-Pack grid sizes.
    atoms
        Atoms object which is used to generate a list of symmetries.
    gamma
        Boolean for whether the k-point sampling should be forced to contain
        the Gamma point through offsets.

    Returns
    -------
        An array with a boolean for each grid size representing whether the
        predicate is satisfied by the k-point sampling.

    """

    pbc_c = atoms.pbc
    symmetries = create_symmetries_object(atoms, symmorphic=False)

    bools = np.zeros(len(mp_grids), dtype=bool)
    for count, size in enumerate(np.array(mp_grids)):
        if gamma:
            offsets = np.array([0.5 / s if s % 2 == 0 and pbc else 0.
                               for s, pbc in zips(size, pbc_c)])
        else:
            offsets = np.array([0., 0., 0.])

        bzpoints = BZPoints(monkhorst_pack(size) + offsets)
        ibzpoints = bzpoints.reduce(symmetries, strict=False,
                                    use_time_reversal=False)
        if -1 not in ibzpoints.bz2bz_Ks:
            bools[count] = True

    return bools


def get_nk_ibz(mp_grids,
               atoms: Atoms,
               gamma: bool):

    pbc_c = atoms.pbc
    symmetries = create_symmetries_object(atoms, symmorphic=False)

    nibz = np.zeros(len(mp_grids), dtype=int)
    for count, size in enumerate(np.array(mp_grids)):
        if gamma:
            offsets = np.array([0.5 / s if s % 2 == 0 and pbc else 0.
                               for s, pbc in zips(size, pbc_c)])
        else:
            offsets = np.array([0., 0., 0.])

        bzpoints = BZPoints(monkhorst_pack(size) + offsets)
        ibzpoints = bzpoints.reduce(symmetries, strict=False,
                                    use_time_reversal=False)
        nibz[count] = len(ibzpoints)

    return nibz


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


def get_bz(calc, comm, pbc_c=np.ones(3, bool)):
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

    return get_reduced_bz(cell_cv, cU_scc, False, comm, pbc_c=pbc_c)


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


def get_reduced_bz(cell_cv, cU_scc, time_reversal, comm,
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
    ibzk_kc = expand_ibz(lU_scc, cU_scc, ibzk_kc, comm, pbc_c=pbc_c)

    # Fold out to full BZ
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               cU_scc.transpose(0, 2, 1))))

    return bzk_kc, ibzk_kc, latibzk_kc


def expand_ibz(lU_scc, cU_scc, latibzk_kc, comm, pbc_c=np.ones(3, bool)):
    """Expand IBZ vertices from lattice group to crystal group.

    Parameters
    ----------
    lU_scc : ndarray
        Lattice symmetry operators.
    cU_scc : ndarray
        Crystal symmetry operators.
    latibzk_kc : ndarray
        Vertices of lattice IBZ.

    Returns
    -------
    ibzk_kc : ndarray
        Vertices of crystal IBZ.

    """

    # Find right cosets. The lattice group is partioned into right cosets of
    # the crystal group. This can in practice be done by testing whether
    # U1 U2^{-1} is in the crystal group as done below.

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
    nibzk_kc = latibzk_kc
    U0_cc = cosets[0][0]  # Origin

    if np.any(~pbc_c):
        nonpbcind = np.argwhere(~pbc_c)

    # Once the cosets are known the irreducible zone is given by picking one
    # operation from each coset. To make sure that the IBZ produced is simply
    # connected we compute the volume of the convex hull of the produced IBZ
    # and pick (one of) the ones that have the smallest volume. This is done by
    # brute force and can sometimes take a while, however, in most cases this
    # is not a problem.
    combs = list(product(*cosets[1:]))[comm.rank::comm.size]
    for U_scc in combs:
        if not len(U_scc):
            continue
        U_scc = np.concatenate([np.array(U_scc), [U0_cc]])
        tmpk_kc = unfold_points(latibzk_kc, U_scc)
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

    volumes = np.zeros(comm.size, float)
    comm.all_gather(volume, volumes)

    minrank = np.argmin(volumes)
    minshape = np.array(ibzk_kc.shape)
    comm.broadcast(minshape, minrank)

    if comm.rank != minrank:
        ibzk_kc = np.zeros(minshape, float)
    comm.broadcast(ibzk_kc, minrank)

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
