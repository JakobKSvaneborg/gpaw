import pytest
import numpy as np
from ase.units import Bohr
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.df import DielectricFunction
from gpaw.test import findpeak


@pytest.mark.response
@pytest.mark.libxc
def test_response_diamond_absorption(in_tmp_dir):
    a = 6.75 * Bohr
    atoms = bulk('C', 'diamond', a=a)

    calc = GPAW(mode='pw',
                kpts=(3, 3, 3),
                eigensolver='rmm-diis',
                occupations=FermiDirac(0.001))

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('C.gpw', 'all')

    eM1_ = 9.727
    eM2_ = 9.548
    w0_ = 10.7782
    I0_ = 5.47
    w_ = 10.7532
    I_ = 5.98

    # Macroscopic dielectric constant calculation
    df = DielectricFunction('C.gpw', frequencies=(0.,), eta=0.001, ecut=50,
                            hilbert=False)
    eM1, eM2 = df.get_macroscopic_dielectric_constant()
    assert eM1 == pytest.approx(eM1_, abs=0.01)
    assert eM2 == pytest.approx(eM2_, abs=0.01)

    # Absorption spectrum calculation RPA
    df = DielectricFunction('C.gpw',
                            eta=0.25,
                            ecut=50,
                            frequencies=np.linspace(0, 24., 241),
                            hilbert=False)
    a0, a = df.get_dielectric_function(filename=None)
    df.check_sum_rule(a.imag)

    assert a0[0].real == pytest.approx(eM1_, abs=0.01)
    assert a[0].real == pytest.approx(eM2_, abs=0.01)
    w, I = findpeak(np.linspace(0, 24., 241), a0.imag)
    assert w == pytest.approx(w0_, abs=0.01)
    assert I / (4 * np.pi) == pytest.approx(I0_, abs=0.1)
    w, I = findpeak(np.linspace(0, 24., 241), a.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I / (4 * np.pi) == pytest.approx(I_, abs=0.1)

    a0, a = df.get_polarizability(filename=None)

    w, I = findpeak(np.linspace(0, 24., 241), a0.imag)
    assert w == pytest.approx(w0_, abs=0.01)
    assert I == pytest.approx(I0_, abs=0.01)
    w, I = findpeak(np.linspace(0, 24., 241), a.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I == pytest.approx(I_, abs=0.01)

    # Absorption spectrum calculation ALDA
    w0_ = 10.7931
    I0_ = 5.36
    w_ = 10.7562
    I_ = 5.8803

    df.get_polarizability(xc='ALDA', filename='ALDA_pol.csv')
    # Here we base the check on a written results file
    df.context.comm.barrier()
    d = np.loadtxt('ALDA_pol.csv', delimiter=',')

    w, I = findpeak(d[:, 0], d[:, 2])
    assert w == pytest.approx(w0_, abs=0.01)
    assert I == pytest.approx(I0_, abs=0.1)
    w, I = findpeak(d[:, 0], d[:, 4])
    assert w == pytest.approx(w_, abs=0.01)
    assert I == pytest.approx(I_, abs=0.1)

    # Absorption spectrum calculation long-range kernel
    w0_ = 10.2189
    I0_ = 5.14
    w_ = 10.2906
    I_ = 5.6955

    a0, a = df.get_polarizability(filename=None, xc='LR0.25')

    w, I = findpeak(np.linspace(0, 24., 241), a0.imag)
    assert w == pytest.approx(w0_, abs=0.01)
    assert I == pytest.approx(I0_, abs=0.1)
    w, I = findpeak(np.linspace(0, 24., 241), a.imag)
    assert w == pytest.approx(w_, abs=0.01)
    assert I == pytest.approx(I_, abs=0.1)

    # Absorption spectrum calculation Bootstrap
    w0_ = 10.37
    I0_ = 5.27
    w_ = 10.4600
    I_ = 6.0263

    a0, a = df.get_polarizability(filename=None, xc='Bootstrap')

    w, I = findpeak(np.linspace(0, 24., 241), a0.imag)
    assert w == pytest.approx(w0_, abs=0.02)
    assert I == pytest.approx(I0_, abs=0.2)
    w, I = findpeak(np.linspace(0, 24., 241), a.imag)
    assert w == pytest.approx(w_, abs=0.02)
    assert I == pytest.approx(I_, abs=0.2)
