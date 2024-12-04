from math import pi

import numpy as np
import pytest
from ase import Atoms
from scipy.special import erf

from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD
from gpaw.core import PWDesc
from gpaw.new.ase_interface import GPAW
from gpaw.new.pw.bloechl_poisson import BloechlPAWPoissonSolver
from gpaw.new.pw.paw_poisson import (SimplePAWPoissonSolver,
                                     SlowPAWPoissonSolver)
from gpaw.new.pw.poisson import PWPoissonSolver


def g(rc, rgd):
    """Gaussian."""
    return rgd.spline(4 / rc**3 / np.pi**0.5 * np.exp(-(rgd.r_g / rc)**2),
                      l=0)


def c(r, rc1, rc2):
    """Coulomb interaction between 2 gaussians."""
    a1 = 1 / rc1**2
    a2 = 1 / rc2**2
    f = 2 * (pi**5 / (a1 + a2))**0.5 / (a1 * a2)
    f *= 16 / pi / rc1**3 / rc2**3
    if r == 0.0:
        return f
    T = a1 * a2 / (a1 + a2) * r**2
    y = 0.5 * f * erf(T**0.5) * (pi / T)**0.5
    return y


def test_psolve():
    """Unit-test for Blöchl's fast Poisson-solver."""
    rgd = RGD(0.01, 500)
    rc1 = 0.6
    rc2 = 0.7
    d12 = 1.35
    g_ai = [[g(rc1, rgd)], [g(rc2, rgd)]]
    v = 7.5
    gcut = 25.0
    pw = PWDesc(gcut=gcut, cell=[2 * v, 2 * v, 2 * v + d12])
    relpos_ac = np.array([[0.5, 0.5, v / (2 * v + d12)],
                          [0.5, 0.5, (v + d12) / (2 * v + d12)]])
    g_aig = pw.atom_centered_functions(g_ai, positions=relpos_ac)
    nt_g = pw.zeros()
    C_ai = g_aig.empty()
    C_ai.data[:] = [0.9, 0.7]
    C_ai.data *= 1.0 / (4.0 * np.pi)**0.5
    g_aig.add_to(nt_g, C_ai)

    charges = [(0.9, rc1, 0.0),
               (0.7, rc2, d12),
               (-0.9, 0.3, 0.0),
               (-0.7, 0.4, d12)]
    e0 = 0.0
    for q1, r1, p1 in charges:
        for q2, r2, p2 in charges:
            d = abs(p1 - p2)
            e12 = 0.5 * q1 * q2 * c(d, r1, r2) / (4 * np.pi)**2
            # print(q1, q2, rc1, rc2, d, e12)
            e0 += e12
    print(e0)

    ps = PWPoissonSolver(pw)
    spps = SimplePAWPoissonSolver(
        pw, [0.3, 0.4], ps, relpos_ac, g_aig.atomdist)
    Q_aL = spps.ghat_aLg.empty()
    Q_aL.data[:] = 0.0
    for a, C_i in C_ai.items():
        Q_aL[a][0] = -C_i[0]
    vt1_g = pw.zeros()
    e1, vHt_g, V1_aL = spps.solve(nt_g, Q_aL, vt1_g)
    F1_av = spps.force_contribution(Q_aL, vHt_g, nt_g)
    assert e1 == pytest.approx(e0, abs=1e-9)
    print('simple', e1, e1 - e0)
    print(spps.force_contribution(Q_aL, vHt_g, nt_g))

    pps = BloechlPAWPoissonSolver(
        pw, [0.3, 0.4], ps, relpos_ac, g_aig.atomdist)
    vt2_g = pw.zeros()
    e2, vHt_g, V2_aL = pps.solve(nt_g, Q_aL, vt2_g)
    F2_av = pps.force_contribution(Q_aL, vHt_g, nt_g)
    assert e2 == pytest.approx(e0, abs=1e-8)
    print('\nfast  ', e2, e2 - e0)
    assert V2_aL.data[::9] == pytest.approx(V1_aL.data[::9], abs=1e-7)
    assert vt2_g.data[:5] == pytest.approx(vt1_g.data[:5], abs=1e-10)
    assert F1_av == pytest.approx(F2_av, abs=3e-6)

    if 0:
        ps = PWPoissonSolver(pw.new(gcut=2 * gcut))
        opps = SlowPAWPoissonSolver(
            pw, [0.3, 0.4], ps, relpos_ac, g_aig.atomdist)
        vt_g = pw.zeros()
        e3, vHt_h, V_aL = opps.solve(nt_g, Q_aL, vt_g)
        print('old   ', e3, e3 - e0)
        print(V_aL.data[::9])
        print(vt_g.data[:5])


def fast_slow(fast):
    atoms = Atoms('H2', [[0, 0, 0], [0.1, 0.2, 0.8]], pbc=True)
    atoms.center(vacuum=3.5)
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 600},
                      poissonsolver={'fast': fast},
                      convergence={'forces': 1e-3},
                      txt=None,
                      symmetry='off')
    atoms.get_potential_energy()
    f = atoms.get_forces()
    eps = 0.001 / 2
    atoms.positions[1, 2] += eps
    ep = atoms.get_potential_energy()
    atoms.positions[1, 2] -= 2 * eps
    em = atoms.get_potential_energy()
    print(f[1, 2], (em - ep) / (2 * eps))


if __name__ == '__main__':
    # test_psolve()
    import sys
    fast_slow(int(sys.argv[1]))
