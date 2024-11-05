import pytest
import numpy as np
from ase.build import bulk, molecule
from ase.parallel import world
from gpaw.gpu import as_np
from gpaw.new.ase_interface import GPAW as GPAW_new

TOL = 1e-8


def build_N2():
    atoms = molecule("N2")
    atoms.center(vacuum=3.0)
    return atoms * (2, 1, 1)


def build_Si():
    atoms = bulk("Si", "diamond", a=5.43)
    return atoms


def get_wfs_coefs(calc):
    ibzwfs = calc.dft.ibzwfs
    xshape = ibzwfs.get_max_shape(global_shape=True)
    dtype = complex if ibzwfs.mode == "pw" else ibzwfs.dtype
    buf_nX = np.empty((ibzwfs.nbands,) + xshape, dtype=dtype)
    coef_sknX = {}
    for spin in range(ibzwfs.nspins):
        coef_sknX[spin] = {}
        for k, rank in enumerate(ibzwfs.rank_k):
            if rank == ibzwfs.kpt_comm.rank:
                wfs = ibzwfs.wfs_qs[ibzwfs.q_k[k]][spin]
                coef_nX = wfs.gather_wave_function_coefficients()
                if coef_nX is not None:
                    coef_nX = as_np(coef_nX)
                    if ibzwfs.mode == "pw":
                        x = coef_nX.shape[-1]
                        if x < xshape[-1]:
                            buf_nX[..., :x] = coef_nX
                            buf_nX[..., x:] = 0.0
                            coef_nX = buf_nX
                    if rank == 0:
                        coef_sknX[spin][k] = coef_nX
                    else:
                        ibzwfs.kpt_comm.send(coef_nX, 0)
            elif ibzwfs.comm.rank == 0:
                ibzwfs.kpt_comm.receive(buf_nX, rank)
                coef_sknX[spin][k] = buf_nX
    return coef_sknX


@pytest.mark.parametrize(
    "name, build_atoms, kwargs",
    [
        ("N2", build_N2, {"mode": "pw", "xc": "LDA", "txt": None}),
        (
            "Si",
            build_Si,
            {"mode": "pw", "xc": "LDA", "kpts": (2, 2, 2), "txt": None},
        ),
    ],
)
def test_write_new_single(name, build_atoms, kwargs):
    atoms = build_atoms()
    calc = GPAW_new(**kwargs)
    atoms.calc = calc
    # write
    atoms.get_potential_energy()
    nt_sR_before = calc.dft.density.nt_sR.gather()
    vt_sR_before = calc.dft.potential.vt_sR.gather()

    coef_sknX_before = get_wfs_coefs(calc)
    calc.write(name + "_calc_all_double.gpw", mode="all")
    calc.write(name + "_calc_all_single.gpw", mode="all", precision="single")
    del calc
    # load
    calc = GPAW_new(name + "_calc_all_single.gpw")
    coef_sknX_after = get_wfs_coefs(calc)
    nt_sR_after = calc.dft.density.nt_sR.gather()
    vt_sR_after = calc.dft.potential.vt_sR.gather()
    if world.rank == 0:
        assert np.allclose(nt_sR_before.data, nt_sR_after.data, atol=TOL)
        assert np.allclose(vt_sR_before.data, vt_sR_after.data, atol=TOL)

        for spin in coef_sknX_before:
            for k in coef_sknX_before[spin]:
                assert np.allclose(
                    coef_sknX_before[spin][k],
                    coef_sknX_after[spin][k],
                    atol=TOL,
                )


if __name__ == "__main__":
    test_write_new_single("N2", build_N2, {"mode": "pw", "xc": "LDA", "txt": None})
    test_write_new_single(
        "Si", build_Si, {"mode": "pw", "xc": "LDA", "kpts": (2, 2, 2), "txt": None}
    )
