import numpy as np
import pytest
from ase.build import bulk
from ase.units import Bohr

from gpaw import FermiDirac
from gpaw.response.df import DielectricFunction, read_response_function
from gpaw.test import findpeak


@pytest.mark.ci
@pytest.mark.dielectricfunction
@pytest.mark.response
@pytest.mark.parametrize('eshift', [None, 4])
@pytest.mark.parametrize('mode', ['pw', 'lcao'])
@pytest.mark.libxc
def test_response_diamond_absorption(in_tmp_dir, eshift, mode, mpi):
    a = 6.75 * Bohr
    atoms = bulk('C', 'diamond', a=a)

    calc = mpi.NewGPAW(
        mode=mode,
        kpts=(3, 3, 3),
        nbands='nao' if mode == 'lcao' else None,
        basis='dzp' if mode == 'lcao' else {},
        eigensolver='rmm-diis' if mode == 'pw' else None,
        occupations=FermiDirac(0.001), txt='out.txt')

    atoms.calc = calc
    atoms.get_potential_energy()
    dft = calc.dft
    if mode != 'pw':
        dft.change_mode('pw')
    dft.write_gpw_file('C.gpw', include_wfs=True)

    if eshift is None:
        eM1_ = 9.727 if mode == 'pw' else 9.4319
        eM2_ = 9.548 if mode == 'pw' else 9.1905
        w0_ = 10.7782 if mode == 'pw' else 10.982
        I0_ = 5.47 if mode == 'pw' else 4.8852
        w_ = 10.7532 if mode == 'pw' else 10.967
        I_ = 5.98 if mode == 'pw' else 5.0459
    else:
        if mode == 'lcao':
            eM1_ = 6.847
            eM2_ = 6.710
            w0_ = 14.982
            I0_ = 4.886
            w_ = 14.967
            I_ = 5.057
        else:
            eM1_ = 6.992
            eM2_ = 6.904
            w0_ = 14.784
            I0_ = 5.47
            w_ = 14.757
            I_ = 5.998

    # Test the old interface to the dielectric constant
    df = DielectricFunction('C.gpw', frequencies=(0.,), eta=0.001, ecut=50,
                            **{'nbands': calc.wfs.bd.nbands}
                            if mode == 'lcao' else {},
                            hilbert=False, eshift=eshift, txt='df.txt',
                            world=mpi.comm)
    eM1, eM2 = df.get_macroscopic_dielectric_constant()
    assert eM1 == pytest.approx(eM1_, abs=0.015)
    assert eM2 == pytest.approx(eM2_, abs=0.01)

    # ----- RPA dielectric function ----- #
    dfcalc = DielectricFunction(
        'C.gpw', eta=0.25, ecut=50,
        frequencies=np.linspace(0, 24., 241), hilbert=False, eshift=eshift,
        world=mpi.comm)
    response = dfcalc.calculate()

    # Test the dielectric constant
    eM1, eM2 = response.dielectric_constant()
    assert eM1 == pytest.approx(eM1_, abs=0.01)
    assert eM2 == pytest.approx(eM2_, abs=0.01)

    # Test the macroscopic dielectric function
    omega_w, eps0M_w, epsM_w = response.dielectric_function().arrays
    w0, I0 = findpeak(omega_w, eps0M_w.imag)
    assert w0 == pytest.approx(w0_, abs=0.01)
    assert I0 / (4 * np.pi) == pytest.approx(I0_, abs=0.1)
    w, I = findpeak(omega_w, epsM_w.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I / (4 * np.pi) == pytest.approx(I_, abs=0.1)

    # Test polarizability
    omega_w, a0rpa_w, arpa_w = response.polarizability().arrays
    w0, I0 = findpeak(omega_w, a0rpa_w.imag)
    assert w0 == pytest.approx(w0_, abs=0.01)
    assert I0 == pytest.approx(I0_, abs=0.01)
    w, I = findpeak(omega_w, arpa_w.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I == pytest.approx(I_, abs=0.01)

    # Test that the bare DF path gives the same macroscopic DF
    # (for untruncated RPA, the bare and inverse paths are equivalent)
    vP_wGG, vchibar_wGG = response._calculate_vchi_symm(
        direction='x', modified=True)
    vchibar_W = response.wblocks.all_gather(vchibar_wGG[:, 0, 0])
    epsM_bare_w = 1. - vchibar_W
    assert epsM_bare_w == pytest.approx(epsM_w, rel=1e-6)

    # ----- TDDFT absorption spectra ----- #

    # Absorption spectrum calculation ALDA
    if eshift is None:
        w_ = 10.7562 if mode == 'pw' else 10.97
        I_ = 5.8803 if mode == 'pw' else 4.915
    else:
        w_ = 14.7615 if mode == 'pw' else 14.9731
        I_ = 5.7946 if mode == 'pw' else 4.9209

    response_alda = dfcalc.calculate(xc='ALDA', rshelmax=0)
    # Here we base the check on a written results file
    response_alda.polarizability().write(filename='ALDA_pol.csv',
                                         comm=mpi.comm)
    dfcalc.context.comm.barrier()
    omega_w, a0alda_w, aalda_w = read_response_function('ALDA_pol.csv')

    assert a0alda_w == pytest.approx(a0rpa_w, rel=1e-4)
    w, I = findpeak(omega_w, aalda_w.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I == pytest.approx(I_, abs=0.1)

    # Absorption spectrum calculation long-range kernel
    if eshift is None:
        w_ = 10.2906 if mode == 'pw' else 10.4213
        I_ = 5.6955 if mode == 'pw' else 5.042
    else:
        w_ = 14.2901 if mode == 'pw' else 14.4245
        I_ = 5.5508 if mode == 'pw' else 4.9553

    response_lr = dfcalc.calculate(xc='LR0.25')
    omega_w, a0lr_w, alr_w = response_lr.polarizability().arrays

    assert a0lr_w == pytest.approx(a0rpa_w, rel=1e-4)
    w, I = findpeak(omega_w, alr_w.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I == pytest.approx(I_, abs=0.1)

    # Absorption spectrum calculation Bootstrap
    if eshift is None:
        w_ = 10.4600 if mode == 'pw' else 10.553
        I_ = 6.0263 if mode == 'pw' else 5.041
    else:
        w_ = 14.2626 if mode == 'pw' else 14.38418
        I_ = 5.3896 if mode == 'pw' else 4.82897

    response_btsr = dfcalc.calculate(xc='Bootstrap')
    omega_w, a0btsr_w, abtsr_w = response_btsr.polarizability().arrays

    assert a0btsr_w == pytest.approx(a0rpa_w, rel=1e-4)
    w, I = findpeak(omega_w, abtsr_w.imag)
    assert w == pytest.approx(w_, abs=0.02)
    assert I == pytest.approx(I_, abs=0.2)

    # import matplotlib.pyplot as plt
    # plt.plot(omega_w, a0rpa_w.imag, label='IP')
    # plt.plot(omega_w, arpa_w.imag, label='RPA')
    # plt.plot(omega_w, aalda_w.imag, label='ALDA')
    # plt.plot(omega_w, alr_w.imag, label='LR0.25')
    # plt.plot(omega_w, abtsr_w.imag, label='Bootstrap')
    # plt.legend()
    # plt.show()
