from gpaw.core import PWDesc
import numpy as np
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD
from gpaw.new.pw.paw_poisson import PAWPoissonSolver, c
from gpaw.new.pw.poisson import PWPoissonSolver


def g(rc, rgd):
    return rgd.spline(4 / rc**3 / np.pi**0.5 * np.exp(-(rgd.r_g / rc)**2),
                      l=0)


def test_psolve():
    rgd = RGD(0.02, 500)
    rc1 = 1.2
    rc2 = 1.7
    d12 = 1.9
    g_ai = [[g(rc1, rgd)], [g(rc2, rgd)]]
    v = 3.5
    pw = PWDesc(gcut=15.0, cell=[2 * v, 2 * v, 2 * v + d12])
    fracpos_ac = np.array([[v, v, v],
                           [v, v, v + d12]])
    g_aig = pw.atom_centered_functions(g_ai, positions=fracpos_ac)
    nt_g = pw.zeros()
    Q_ai = g_aig.empty()
    Q_ai.data[:] = [0.9, 0.7]
    Q_ai.data *= 1.0 / (4.0 * np.pi)**0.5
    g_aig.add_to(nt_g, Q_ai)
    print(nt_g.integrate())
    ps = PWPoissonSolver(pw)
    pps = PAWPoissonSolver(pw, [rc1, rc2], ps, fracpos_ac, g_aig.atomdist)
    Q_ai.data *= 1.0 / (4.0 * np.pi)**0.5
    vt_g = pw.zeros()
    e, vHt_g, V_aL = pps.solve(nt_g, Q_ai, vt_g)
    print(e, V_aL)


if __name__ == '__main__':
    rc1 = 0.1
    rc2 = 0.2
    print(c(8.0, rc1, rc2) *
          4 / rc1**3 / np.pi**0.5 *
          4 / rc2**3 / np.pi**0.5 / (4 * np.pi)**2)
    test_psolve()
