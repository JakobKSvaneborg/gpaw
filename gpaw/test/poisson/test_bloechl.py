import numpy as np
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD
from gpaw.core import PWDesc
from gpaw.new.pw.bloechl_poisson import BloechlPAWPoissonSolver, c
from gpaw.new.pw.paw_poisson import (SimplePAWPoissonSolver,
                                     SlowPAWPoissonSolver)
from gpaw.new.pw.poisson import PWPoissonSolver
from gpaw.new.ase_interface import GPAW
from ase import Atoms


def g(rc, rgd):
    return rgd.spline(4 / rc**3 / np.pi**0.5 * np.exp(-(rgd.r_g / rc)**2),
                      l=0)


def test_psolve():
    """Unit-test for Blöchl's fast Poisson-solver (WIP)."""
    rgd = RGD(0.01, 500)
    rc1 = 0.6
    rc2 = 0.7
    d12 = 1.35
    g_ai = [[g(rc1, rgd)], [g(rc2, rgd)]]
    v = 7.5
    gcut=25
    pw = PWDesc(gcut=gcut, cell=[2 * v, 2 * v, 2 * v + d12])
    fracpos_ac = np.array([[0.5, 0.5, v / (2 * v + d12)],
                           [0.5, 0.5, (v + d12) / (2 * v + d12)]])
    g_aig = pw.atom_centered_functions(g_ai, positions=fracpos_ac)
    nt_g = pw.zeros()
    C_ai = g_aig.empty()
    C_ai.data[:] = [0.9, 0.7]
    C_ai.data *= 1.0 / (4.0 * np.pi)**0.5
    g_aig.add_to(nt_g, C_ai)
    # print(nt_g.integrate())

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
        pw, [0.3, 0.4], ps, fracpos_ac, g_aig.atomdist)
    Q_aL = spps.ghat_aLg.empty()
    Q_aL.data[:] = 0.0
    for a, C_i in C_ai.items():
        Q_aL[a][0] = -C_i[0]
    vt_g = pw.zeros()
    e1, vHt_g, V_aL = spps.solve(nt_g, Q_aL, vt_g)
    print('simple', e1, e1 - e0)
    print(V_aL.data[::9])
    print(vt_g.data[:5])

    pps = BloechlPAWPoissonSolver(
        pw, [0.3, 0.4], ps, fracpos_ac, g_aig.atomdist)
    vt_g = pw.zeros()
    e2, vHt_g, V_aL = pps.solve(nt_g, Q_aL, vt_g)
    print('fast  ', e2, e2 - e0)
    print(V_aL.data[::9])
    print(vt_g.data[:5])

    charges = [(0.9, rc1, 0.0),
               (0.7, rc2, d12),
               (-0.9, 0.8, 0.0),
               (-0.7, 0.8, d12)]
    e20 = 0.0
    for q1, r1, p1 in charges:
        for q2, r2, p2 in charges:
            d = abs(p1 - p2)
            e12 = 0.5 * q1 * q2 * c(d, r1, r2) / (4 * np.pi)**2
            # print(q1, q2, rc1, rc2, d, e12)
            e20 += e12

    if 0:
        ps = PWPoissonSolver(pw.new(gcut=2 * gcut))
        opps = SlowPAWPoissonSolver(
            pw, [0.3, 0.4], ps, fracpos_ac, g_aig.atomdist)
        vt_g = pw.zeros()
        e3, vHt_h, V_aL = opps.solve(nt_g, Q_aL, vt_g)
        print('old   ', e3, e3 - e0)
        print(V_aL.data[::9])
        print(vt_g.data[:5])


def test_fast_slow(fast):
    atoms = Atoms('H2', [[0, 0, 0], [0.1, 0.2, 0.7]], pbc=True)
    atoms.center(vacuum=3.5)
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 800},
                      poissonsolver={'fast': fast})
    atoms.get_potential_energy()


if __name__ == '__main__':
    test_psolve()
    import sys
    # test_fast_slow(int(sys.argv[1]))
