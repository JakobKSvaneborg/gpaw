import pytest
from ase.build import molecule
from gpaw.dft import DFT


def test_xc():

    etot_hse = -7.085318

    atoms = molecule('H2', cell=[7, 7, 7])
    atoms.center()
    atoms.set_pbc(True)

    ppcg = {'name': 'ppcg',
            'niter': 30,
            'min_niter': 2}

    params = {'xc': 'PBE',
              'mode': {'name': 'pw', 'ecut': 400},
              'kpts': {'size': [1, 1, 1],
                       'gamma': True},
              'eigensolver': ppcg,
              'nbands': 4,
              'mixer': {'method': 'fullspin',
                        'backend': 'fft',
                        'beta': 0.05,
                        'nmaxold': 7,
                        'weight': 50.0}}

    xc_hse = {'name': 'HYB_GGA_XC_HSE06',
              'fraction': 0.26,
              'omega': 0.11,
              'backend': 'pw'}

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge()

    dft.change_xc(xc_hse)
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
