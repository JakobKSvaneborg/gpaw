import pytest
from ase.build import molecule
from gpaw import GPAW
from gpaw.test import gen
from gpaw.xas import XAS, get_oscillator_strength
import gpaw.mpi as mpi


def folding_is_normalized(xas: XAS, dks, rel: float = 1e-5) -> bool:
    xs, ys_cn = xas.get_oscillator_strength(dks=dks)
    ys_summed_c = ys_cn.sum(axis=1)
    xf, yf_cn = xas.get_spectra(fwhm=0.5, dks=dks)
    dxf = xf[1:] - xf[:-1]
    assert dxf == pytest.approx(dxf[0])
    yf_summed_c = yf_cn.sum(axis=1) * dxf[0]

    return yf_summed_c == pytest.approx(ys_summed_c, rel=rel)


@pytest.fixture
def s1s1ch_name():
    setupname = 'S1s1ch'
    gen('S', name=setupname, corehole=(1, 0, 1), gpernode=30, write_xml=True)
    return setupname


@pytest.fixture
def s2p1ch_name():
    setupname = 'S2p1ch'
    gen('S', name=setupname, corehole=(2, 1, 1), gpernode=30, write_xml=True)
    return setupname


def test_sulphur_2p_spin_io(in_tmp_dir, add_cwd_to_setup_paths, s2p1ch_name):
    """Make sure this calculation does not fail
    because of get_spin_contamination"""
    atoms = molecule('SH2')
    atoms.center(3)

    atoms.set_initial_magnetic_moments([1, 0, 0])
    atoms.calc = GPAW(mode='fd', h=0.3, spinpol=True,
                      setups={'S': s2p1ch_name}, txt=None,
                      convergence={
                          'energy': 0.1, 'density': 0.1, 'eigenstates': 0.1})
    atoms.get_potential_energy()


def test_sulphur_1s_xas(in_tmp_dir, add_cwd_to_setup_paths, s1s1ch_name):
    atoms = molecule('SH2')
    atoms.center(3)

    setupname = s1s1ch_name

    nbands = 6
    nocc = 4  # for SH2
    atoms.calc = GPAW(mode='fd', h=0.3, nbands=nbands,
                      setups={'S': setupname}, txt=None)
    atoms.get_potential_energy()

    dks = 20
    xas = XAS(atoms.calc)
    x, y_cn = xas.get_oscillator_strength(dks=dks)
    assert y_cn.shape == (3, nbands - nocc)
    assert x[0] == dks
    assert xas.nocc == nocc

    assert folding_is_normalized(xas, dks)

    atoms.calc = GPAW(mode='fd', h=0.3, nbands=nbands,
                      setups={'S': setupname}, txt=None,
                      charge=-1)
    atoms[0].magmom = 1
    atoms.get_potential_energy()

    xas = XAS(atoms.calc, nocc_cor=-1)
    x, y_cn = xas.get_oscillator_strength(dks=dks)
    assert xas.nocc == nocc
    assert y_cn.shape == (3, nbands - nocc)
    assert x[0] == dks
    assert folding_is_normalized(xas, dks)


def test_sulphur_2p_xas(in_tmp_dir, add_cwd_to_setup_paths, s2p1ch_name):
    atoms = molecule('SH2')
    atoms.center(3)
    dks = 20
    atoms.calc = GPAW(mode='fd', h=0.3, setups={'S': s2p1ch_name}, txt=None)
    atoms.get_potential_energy()

    xas = XAS(atoms.calc)
    assert folding_is_normalized(xas, dks)


def test_lean_io(in_tmp_dir, add_cwd_to_setup_paths, s1s1ch_name):
    atoms = molecule('SH2')
    atoms.center(3)

    nbands = 6
    atoms.calc = GPAW(mode='fd', h=0.3, nbands=nbands,
                      setups={'S': s1s1ch_name}, txt=None)
    atoms.get_potential_energy()

    dks = 20
    xas0 = XAS(atoms.calc)
    mefname = 'me.dat.npz'
    xas0.write(mefname)
    x0, y0_cn = xas0.get_oscillator_strength(dks=dks)

    x1, y1_cn = get_oscillator_strength(mefname, dks=dks)
    assert x1 == pytest.approx(x0)
    assert y1_cn == pytest.approx(y0_cn)


def test_parallel(in_tmp_dir, add_cwd_to_setup_paths, s2p1ch_name):
    print('#### size: ', mpi.world.size, mpi.size)
    if 0 and mpi.world.size < 2:
        return

    atoms = molecule('SH2')
    atoms.center(3)

    # serial calculation
    fserial = 'serial_xas.npz'
    comm = mpi.world.new_communicator([mpi.world.rank])
    print('serial, rank, size:', mpi.world.rank, comm.size)
    atoms.calc = GPAW(mode='fd', h=0.3, setups={'S': s2p1ch_name},
                      txt=None, communicator=comm)
    atoms.get_potential_energy()
    print('serial, atoms.calc.world.size:', atoms.calc.world.size)
    xas = XAS(atoms.calc)
    xas.write(fserial)

    # parallel calculation
    fparallel = 'serial_xas.npz'
    atoms.calc = GPAW(mode='fd', h=0.3, setups={'S': s2p1ch_name}, txt=None)
    atoms.get_potential_energy()
    print('parallel, atoms.calc.world.size:', atoms.calc.world.size)
    xas.write(fparallel)

    dks = 20
    xs, ys = get_oscillator_strength(fserial, dks=dks)
    xp, yp = get_oscillator_strength(fparallel, dks=dks)
    assert (xs == xp).all()
    assert (ys == yp).all()
