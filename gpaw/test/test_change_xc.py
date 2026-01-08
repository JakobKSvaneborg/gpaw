import pytest
from ase.build import molecule
from gpaw.dft import DFT


def test_xc():

    etot_hse = -10.014548

    atoms = molecule('H2', cell=[4, 4, 4])
    atoms.center()
    atoms.set_pbc(True)

    ppcg = {'name': 'ppcg',
            'niter': 30,
            'min_niter': 2}

    params = {'xc': 'PBE',
              'mode': {'name': 'pw', 'ecut': 400},
              'nbands': 3,
              'eigensolver': ppcg,
              'convergence': {'eigenstates': 1e-4, 'density': 1e-2}}

    xc_hse = {'name': 'HYB_GGA_XC_HSE06',
              'fraction': 0.26,
              'omega': 0.11,
              'backend': 'pw'}

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge()

    dft.change_xc(xc_hse)

    # fixed_density
    if 0:
        dft.scf_loop.update_density_and_potential = False
        dft.converge(steps=1)
        dft.scf_loop.update_density_and_potential = True

    ase_calc = dft.ase_calculator()
    etot_xc = ase_calc.get_potential_energy(atoms)

    if 0:
        # check against HSE
        from gpaw.new.ase_interface import GPAW
        params['xc'] = xc_hse
        calc = GPAW(**params)
        atoms.calc = calc
        etot_hse = atoms.get_potential_energy()

    assert etot_xc == pytest.approx(etot_hse)


if __name__ == "__main__":
    test_xc()
