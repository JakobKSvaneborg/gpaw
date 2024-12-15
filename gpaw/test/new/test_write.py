import pytest
from ase.build import bulk, molecule
from gpaw.new.ase_interface import GPAW


def build_N2():
    atoms = molecule("N2")
    atoms.center(vacuum=3.0)
    return atoms * (2, 1, 1)


def build_Si():
    atoms = bulk("Si", "diamond", a=5.43)
    return atoms


@pytest.mark.parametrize(
    "name, build_atoms, kwargs",
    [
        ("N2", build_N2, {"mode": "pw", "txt": None}),
        ("Si", build_Si, {"mode": "pw", "kpts": (2, 2, 2), "txt": None}),
        ("N2", build_N2, {"mode": "fd", "txt": None}),
        ("Si", build_Si, {"mode": "fd", "kpts": (2, 2, 2), "txt": None}),
        ("N2", build_N2, {"mode": "lcao", "txt": None}),
        ("Si", build_Si, {"mode": "lcao", "kpts": (2, 2, 2), "txt": None}),
    ],
)
def test_write_new_single(name, build_atoms, kwargs):
    atoms = build_atoms()
    calc = GPAW(**kwargs)
    atoms.calc = calc
    atoms.get_potential_energy()
    # write
    mode = kwargs["mode"]
    gpw_name = f"{name}_{mode}_calc_all.gpw"

    calc.write(gpw_name + "_double", mode="all", precision="double")
    calc.write(gpw_name + "_single", mode="all", precision="single")
    dft1 = calc.dft
    # load
    calc = GPAW(gpw_name + "_single")
    dft2 = calc.dft
    # pytest.approx by default takes the relative tolerance of 1e-6,
    # the test should be 1e-8?
    assert dft1.density.nt_sR.data == pytest.approx(dft2.density.nt_sR.data)
    assert dft1.potential.vt_sR.data == pytest.approx(
        dft2.potential.vt_sR.data)
    for wfs1, wfs2 in zip(dft1.ibzwfs, dft2.ibzwfs):
        if mode == 'lcao':
            assert wfs1.C_nM.data == pytest.approx(wfs2.C_nM.data)
        else:
            assert wfs1.psit_nX.data == pytest.approx(wfs2.psit_nX.data)
