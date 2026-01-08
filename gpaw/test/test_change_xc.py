from ase.build import molecule
from gpaw.new.ase_interface import GPAW

def test_xc():

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
              'spinpol': True,
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

    xc_pbe = {'name': 'PBE'}
    xc_lda = {'name': 'LDA'}

    calc = GPAW(**params)
    atoms.calc = calc
    etot_pbe = atoms.get_potential_energy()

    params = calc.params
    calc.dft.change_xc(atoms, params, xc_lda)
    etot_xc = atoms.get_potential_energy()
    print('etot_pbe=', etot_pbe)
    print('etot_xc=', etot_xc)

if __name__ == "__main__":
    test_xc()
