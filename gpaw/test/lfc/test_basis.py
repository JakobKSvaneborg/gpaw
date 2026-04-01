from gpaw.core import UGArray, UGDesc
from gpaw.lfc import BasisFunctions
from gpaw.new.basis_functions import (
    BasisFunctionDesc, BasisFunctionCollectionBase, LFCSystemDesc)

from gpaw.new.basis_functions_purepython \
    import BasisFunctionCollectionPurePython
from gpaw.spline import Spline
from gpaw.gpu import as_np
# from gpaw import GPAW_NO_C_EXTENSION
from gpaw.mpi import world, MPIComm

import numpy as np
import pytest
import copy
from collections import namedtuple
from typing import cast


def xp_params_no_cpupy():
    """Use as @pytest.mark.parametrize("xp", xp_params_no_cpupy())
    to run both xp=np and xp=cp cases, but skipping fake Cupy.
    """
    # return [
    #     pytest.param(np, id="numpy"),
    #     pytest.param(
    #         cp,
    #         id="cupy",
    #         marks=[
    #             pytest.mark.gpu,
    #             pytest.mark.skipif(cupy_is_fake, reason="Fake Cupy"),
    #         ],
    #     ),
    # ]
    # WIP: doesn't work with Cupy ATM
    return [pytest.param(np, id="numpy")]


def parametrize_purepython():
    """Use as @pytest.mark.parametrize("purepython", parametrize_purepython())
    to run both purepython and non-purepython versions, skipping the latter if
    not available.
    """
    return [True]
    # return [True,
    #         pytest.param(
    #             False,
    #             marks=[pytest.mark.skipif(GPAW_NO_C_EXTENSION,
    #                                       reason="No C extension")]
    #         )]


def make_random_phi(
        l: int,
        cutoff: float,
        rng: np.random.Generator,
        num_points: int = 100) -> BasisFunctionDesc:
    """"""
    values = rng.random(size=num_points)
    # last value must be 0
    values[-1] = 0.0
    return BasisFunctionDesc(l, cutoff, values)


# Generate a bunch of different grid fixtures:
GridShape = namedtuple("GridShape", ["cell", "size"])
# Can (must?) separately specify periodic and zero-boundaries
BoundaryConds = namedtuple("BoundaryConds", ["pbc", "zerobc"])


@pytest.fixture(params=[
    pytest.param(
        GridShape(cell=[3, 3, 3], size=(14, 14, 14)),
        id="UniformOrthorhombic",),
    pytest.param(
        GridShape(cell=[1, 2, 3], size=(16, 10, 11)),
        id="NonUniformOrthorombic",
        marks=pytest.mark.ci),
    pytest.param(
        GridShape(cell=[3, 2, 3], size=(5, 8, 9)),
        id="SmallOrthorhombic"),
    pytest.param(
        GridShape(cell=[[0, 1, 1], [3, 0, 3], [2, 2, 0]], size=(16, 8, 7)),
        id="WeirdShape"),
    pytest.param(
        GridShape(cell=[[1, 2, 3], [0, 0, 3], [2, 2, 0]], size=(16, 10, 7)),
        id="VeryWeirdShape",
        marks=pytest.mark.slow
    )],
    scope="module")
def fixt_grid_shape(request) -> GridShape:
    """Parametrized grid shape fixture, generates grids of different shapes
    and sizes."""
    return request.param


@pytest.fixture(params=[
    pytest.param(
        BoundaryConds(pbc=[True, True, True], zerobc=[False, False, False]),
        id="AllPeriodic"),
    pytest.param(
        BoundaryConds(pbc=[False, True, True], zerobc=[True, False, False]),
        id="SomePeriodic"),
    pytest.param(
        BoundaryConds(pbc=[False, False, False], zerobc=[True, True, True]),
        id="AllNonPeriodic")],
    scope="module")
def fixt_bc(request) -> BoundaryConds:
    """Generates bunch of different boundary conditions for grids.
    TODO: test cases combinations like "pbc = False and zerobc = False.
    This wouldn't pass tests currently because we're comparing against the old
    LFC code, which only has pbc (which is actually ~zerobc)"""
    return request.param


@pytest.fixture(scope="module", params=[
    pytest.param(None, id="NoBlocking", marks=pytest.mark.slow),
    pytest.param(8, id="Block8"),
    pytest.param([5, 6, 7], id="Block567")
])
def fixt_block_size(request) -> int | tuple[int, int, int] | None:
    """"""
    return request.param


@pytest.fixture(scope="module")
def fixt_lfc_system(fixt_grid_shape, fixt_bc) -> LFCSystemDesc:
    """"""

    # Grid does automatic domain decomp if running with MPI
    global world
    world = cast(MPIComm, world)

    grid_def: GridShape = fixt_grid_shape

    grid = UGDesc(cell=grid_def.cell, size=grid_def.size, comm=world,
                  pbc=fixt_bc.pbc, zerobc=fixt_bc.zerobc)

    # helper for scaling radial cutoffs: length of the longest cell vector
    a = float(np.max(np.linalg.norm([grid.cell_cv[c] for c in range(3)])))

    rng = np.random.default_rng(404)

    # Make system of 3 atoms, 2 of which are identical (same basis funcs).
    # Also give the other atom type a different number of basis funcs
    phi_lists = []
    for i in range(2):
        s = make_random_phi(0, 0.78 * a, rng, num_points=100)
        p = make_random_phi(1, 1.15 * a, rng, num_points=100)
        d = make_random_phi(2, 0.1 * a, rng, num_points=100)

        funcs = [s, p, d] if i == 0 else [s, d]
        phi_lists.append(funcs)

    relpos_ac = np.asarray([(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]])

    phi_aj = []
    for a in range(3):
        phi_list = phi_lists[0] if a < 2 else phi_lists[1]
        phi_aj.append(phi_list)

    return LFCSystemDesc(grid, phi_aj, relpos_ac)


def make_legacy_basis_functions(lfc: BasisFunctionCollectionBase, xp) \
        -> BasisFunctions:
    """Build legacy basis functions (Spline objects) and the actual
    BasisFunctions class, for comparison with new code."""

    phi_aj: list[list[Spline]] = []

    for a in range(lfc.num_atoms):
        phi_datas = lfc.atom_static_data.phi_aj[a]
        splines = [Spline.from_data(phi.l, phi.cutoff, phi.f_r)
                   for phi in phi_datas]
        phi_aj.append(splines)

    # DON'T use grid._gd, it gets its pbc fixed as ~zerobc
    from gpaw.old.grid_descriptor import GridDescriptor
    legacy_grid = GridDescriptor(N_c=lfc.grid.size, cell_cv=lfc.grid.cell_cv,
                                 comm=lfc.grid.comm, pbc_c=lfc.grid.pbc_c)

    assert np.all(legacy_grid.pbc_c == lfc.grid.pbc_c)

    # cut=True means non-periodic boundaries don't let basis funcs leak into
    # other cells. With False such leaks would throw GridBoundsError.
    # Old BasisFunctions has False as default, but apparently it's not really
    # used because basis.py creates the object with True anyway.
    basis = BasisFunctions(
        legacy_grid,
        phi_aj,
        xp=xp,
        cut=True)

    relpos_ac = lfc.get_atom_positions(grid_relative=True)
    basis.set_positions(relpos_ac)
    return basis


def make_basis(system: LFCSystemDesc,
               xp,
               purepython: bool,
               block_size: int | None) -> BasisFunctionCollectionBase:
    """"""
    if purepython:
        basis = BasisFunctionCollectionPurePython(
            system,
            use_gpu=(xp is not np),
            block_size=block_size)
    else:
        raise NotImplementedError("WIP")
        # from gpaw.new.basis_functions_cpp import BasisFunctionCollection
        # basis = BasisFunctionCollection(
        #     system,
        #     use_gpu = (xp is not np),
        #     block_size=block_size
        # )

    return basis


@pytest.mark.parametrize("xp", xp_params_no_cpupy())
@pytest.mark.parametrize("purepython", parametrize_purepython())
def test_basis_creation(fixt_lfc_system: LFCSystemDesc, xp, purepython: bool,
                        fixt_block_size):
    """"""
    if not purepython and xp is np:
        pytest.skip(reason="CPU + C++ is WIP")

    system = fixt_lfc_system
    basis = make_basis(system, xp, purepython, fixt_block_size)

    assert basis.uses_gpu() == (xp is not np)
    assert basis.num_atoms == len(system.relpos_ac)

    # check that there are no duplicate phi
    all_phi = basis.get_phi_instances()
    has_duplicate_phi = len({id(phi) for phi in all_phi}) != len(all_phi)
    assert not has_duplicate_phi

    # If all boundaries are non-periodic, shouldn't have any "image" phis
    if np.all(~system.grid.pbc_c):
        num_phi_expected = sum(len(phi_j) for phi_j in system.phi_aj)
        assert len(basis.get_phi_instances()) == num_phi_expected


@pytest.mark.skipif(world.size > 1, reason="TODO, probably")
@pytest.mark.parametrize("xp", xp_params_no_cpupy())
def test_no_blocking(fixt_lfc_system: LFCSystemDesc, xp):
    """Tests that blocking = None makes sense. Useful because under the
    hood the same blocking logic is still ran but with block_size == grid_size
    """

    basis = make_basis(fixt_lfc_system, xp, purepython=True, block_size=None)
    block_to_phi = basis.get_block_to_phi_map()

    # Should have only the (0, 0, 0) block
    # TODO mpi? some domain may be empty
    assert len(block_to_phi) == 1
    assert (0, 0, 0) in block_to_phi.keys()
    block_coords = (0, 0, 0)

    # Check block index ranges
    np.testing.assert_equal(basis.geometry.block_start_Bc[block_coords],
                            basis.grid.start_c)
    np.testing.assert_equal(basis.geometry.block_end_Bc[block_coords],
                            basis.grid.end_c)

    # check that there are no duplicate phi in the block
    block_phi_j = block_to_phi[block_coords]
    has_duplicates = len({id(phi) for phi in block_phi_j}) != len(block_phi_j)
    assert not has_duplicates

    # The block should have overlap with all phi
    all_phi = basis.get_phi_instances()
    assert len(block_phi_j) == len(all_phi)
    # technically this is only airtight if we also check that all_phi has no
    # duplicates. But that is done in test_basis_creation()


@pytest.mark.parametrize("xp", xp_params_no_cpupy())
@pytest.mark.parametrize("purepython", parametrize_purepython())
@pytest.mark.parametrize("row_range", [None, range(1, 4), range(0, 100000)])
def test_potential_matrix(
    fixt_lfc_system: LFCSystemDesc,
    xp,
    purepython: bool,
    fixt_block_size,
    row_range: range | None
):
    """"""
    if not purepython and xp is np:
        pytest.skip(reason="CPU + C++ is WIP")

    system = fixt_lfc_system
    basis = make_basis(system, xp, purepython, fixt_block_size)

    if row_range:
        assert row_range.step == 1
        basis.set_matrix_distribution(row_range.start, row_range.stop)

    # Test array that we sandwich between basis funcs and integrate
    vt_G: UGArray = system.grid.zeros(xp=xp)

    rng = xp.random.default_rng(222)
    noise = rng.random(vt_G.data.shape)
    vt_G.data += noise

    V_MN = basis.calculate_potential_matrix(vt_G.data)

    # Repeat with old code and compare. Need to manually create output array,
    # and I guess need to also manually clamp row_range
    basis_old = make_legacy_basis_functions(basis, xp=xp)
    if row_range:
        clamped_row_range = range(max(row_range.start, basis.mu_range.start),
                                  min(row_range.stop, basis.mu_range.stop))
        basis_old.set_matrix_distribution(clamped_row_range.start,
                                          clamped_row_range.stop)
        out_shape = (clamped_row_range.stop - clamped_row_range.start,
                     basis_old.Mmax)
    else:
        out_shape = (basis_old.Mmax, basis_old.Mmax)

    V_MN_ref = xp.zeros(out_shape)
    basis_old.calculate_potential_matrix(vt_G.data, V_MN_ref, 0)

    """Old code does include some elements in the upper matrix triangle,
    I guess this is done to avoid complex branching.
    Remove them here, while accounting for possible row distribution in which
    case we need to carefully pick which "diagonal" to use in xp.tril(...)
    """
    diagonal = 0 if row_range is None else row_range.start
    V_MN_ref = xp.tril(V_MN_ref, k=diagonal)

    xp.testing.assert_allclose(V_MN,
                               V_MN_ref,
                               rtol=1e-10,
                               atol=1e-12)

    # Test also using the optional 'out' argument
    V_MN_out = xp.empty_like(V_MN)
    dummy_out = basis.calculate_potential_matrix(vt_G.data, out=V_MN_out)
    assert dummy_out is V_MN_out

    xp.testing.assert_allclose(V_MN,
                               V_MN_out,
                               rtol=1e-10,
                               atol=1e-12)


@pytest.mark.parametrize("xp", xp_params_no_cpupy())
@pytest.mark.parametrize("purepython", parametrize_purepython())
@pytest.mark.parametrize("num_spins", [1, 2])
def test_add_to_density(
    fixt_lfc_system: LFCSystemDesc,
    xp,
    purepython: bool,
    num_spins: int,
    fixt_block_size
):
    """"""

    if not purepython and xp is np:
        pytest.skip(reason="CPU + C++ is WIP")

    system = fixt_lfc_system
    basis = make_basis(system, xp, purepython, fixt_block_size)
    # Matrix distribution does not matter here.
    # TODO separate test that it indeed does not matter?

    # Test density
    rng = xp.random.default_rng(432)
    sG_shape = (num_spins, *system.grid.mysize_c)
    nt_sG = xp.empty(sG_shape)
    nt_sG[:] = 0.5

    nt_sG_ref = nt_sG.copy()

    # Make test "array" f_asi. This is actually a dict of arrays, because
    # the number of basis functions (range of i) can vary per atom.
    f_asi = {}
    for atom_idx in range(basis.num_atoms):
        shape_si = (num_spins, basis.get_num_phi_a(atom_idx))
        f_asi[atom_idx] = rng.random(shape_si)

    basis.add_to_density(nt_sG, f_asi)

    # Repeat with old code.
    # this still only works with numpy (density is converted internally)
    f_asi = {a: as_np(f_asi[a]) for a in f_asi.keys()}
    basis_old = make_legacy_basis_functions(basis, xp=np)
    basis_old.add_to_density(nt_sG_ref, f_asi)

    xp.testing.assert_allclose(nt_sG, nt_sG_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("xp", xp_params_no_cpupy())
@pytest.mark.parametrize("purepython", parametrize_purepython())
@pytest.mark.parametrize("num_spins", [1, 2])
@pytest.mark.skipif(world.size < 2, reason="Not enough MPI ranks")
def test_domain_decomposition(
    fixt_lfc_system: LFCSystemDesc,
    xp,
    purepython: bool,
    num_spins: int,
    fixt_block_size
):
    """"""
    if not purepython and xp is np:
        pytest.skip(reason="CPU + C++ is WIP")

    system = fixt_lfc_system
    basis = make_basis(system, xp, purepython, fixt_block_size)

    global world
    world = cast(MPIComm, world)
    # Did we construct with the right comm
    assert system.grid.comm is world

    serial_comm = world.new_communicator([0])

    serial_system = copy.copy(system)
    serial_system.grid = system.grid.new(comm=serial_comm)
    serial_basis = make_basis(serial_system, xp, purepython, fixt_block_size)

    assert serial_basis.num_atoms == basis.num_atoms
    assert serial_basis.mu_range == basis.mu_range

    # add_to_density. We want to compare against a pure-serial computation,
    # so each rank should use the same f_asi input => use same seed
    rng = xp.random.default_rng(999)
    f_asi = {}
    for atom_idx in range(basis.num_atoms):
        shape_si = (num_spins, basis.get_num_phi_a(atom_idx))
        f_asi[atom_idx] = rng.random(shape_si)

    """Goal is to test that add_to_density on one rank (full grid) is
    equivalent to add_to_density on the domain-decomposed grid. We need to
    generate a random input density array on the full grid, then decompose it.
    Easiest is to generate it on all ranks with same seed, then just copy
    correct parts of it as the decomposed input. This avoids need for scatterv
    or similar.
    """

    # FIXME: mpi_gather and mpi_scatter crash or deadlock if the nt_sG arrays
    # are not same shape on all ranks. Such calls are invalid but we should
    # raise exception, not crash/deadlock!

    # Re-seed just in case
    rng = xp.random.default_rng(841)

    # Grid zerobc complications:
    # - First grid point is implicitly deleted
    # - mysize_c is aware of this, so is global_shape(), but size is not
    # - start_c starts from 1 in the first MPI rank
    # - Data arrays are smaller in first rank
    # UGArray and UGDesc.empty, UGDesc.zeros are aware of some of this, but
    # manual array ops are still complicated because of start_c.

    shape_sR = (num_spins, *serial_basis.grid.global_shape())
    nt_sR_serial = rng.random(shape_sR)
    # Decomposed density array:
    shape_decomp_sR = (num_spins, *basis.grid.myshape)
    nt_sR = xp.empty(shape_decomp_sR)

    def manual_scatter(full_nt_sR, decomp_nt_sR):
        start_c = system.grid.start_c.copy()
        end_c = system.grid.end_c.copy()
        # handle first grid point missing if using zerobc
        zerobc = system.grid.zerobc_c
        start_c[zerobc] -= 1
        end_c[zerobc] -= 1

        decomp_nt_sR[:] = full_nt_sR[:,
                                     start_c[0]:end_c[0],
                                     start_c[1]:end_c[1],
                                     start_c[2]:end_c[2]]

    manual_scatter(nt_sR_serial, nt_sR)

    basis.add_to_density(nt_sR, f_asi)
    # Do the serial part separately in all ranks because this test doesn't
    # actually use MPI for scattering the reference result
    serial_basis.add_to_density(nt_sR_serial, f_asi)

    nt_sG_ref = xp.empty_like(nt_sR)
    # Scatter the serial result back to all ranks
    manual_scatter(nt_sR_serial, nt_sG_ref)

    xp.testing.assert_allclose(nt_sR, nt_sG_ref, rtol=1e-10, atol=1e-12)
