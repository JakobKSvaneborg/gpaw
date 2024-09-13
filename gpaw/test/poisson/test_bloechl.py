from gpaw.core import PWDesc
import numpy as np
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD
from gpaw.new.pw.paw_poisson import PAWPoissonSolver, c, SimplePAWPoissonSolver
from gpaw.new.pw.poisson import PWPoissonSolver


def g(rc, rgd):
    return rgd.spline(4 / rc**3 / np.pi**0.5 * np.exp(-(rgd.r_g / rc)**2),
                      l=0)


def test_psolve():
    rgd = RGD(0.02, 500)
    rc1 = 0.6
    rc2 = 0.7
    d12 = 3.6
    g_ai = [[g(rc1, rgd)], [g(rc2, rgd)]]
    v = 5.5
    pw = PWDesc(gcut=15.0, cell=[2 * v, 2 * v, 2 * v + d12])
    fracpos_ac = np.array([[0.5, 0.5, v / (2 * v + d12)],
                           [0.5, 0.5, (v + d12) / (2 * v + d12)]])
    g_aig = pw.atom_centered_functions(g_ai, positions=fracpos_ac)
    nt_g = pw.zeros()
    Q_ai = g_aig.empty()
    Q_ai.data[:] = [0.9, -0.9]
    Q_ai.data *= 1.0 / (4.0 * np.pi)**0.5
    g_aig.add_to(nt_g, Q_ai)
    print(nt_g.integrate())
    if 1:
        ps = PWPoissonSolver(pw)
        spps = SimplePAWPoissonSolver(
            pw, [0.3, 0.4], ps, fracpos_ac, g_aig.atomdist)
        Q_ai.data *= 0
        vt_g = pw.zeros()
        e, vHt_g, V_aL = spps.solve(nt_g, Q_ai, vt_g)
        print(e, V_aL)
    grid = pw.uniform_grid_with_grid_spacing(grid_spacing=0.1)
    v_R = vHt_g.ifft(grid=grid)
    nt_R = nt_g.ifft(grid=grid)
    print(grid)
    import matplotlib.pyplot as plt
    n = grid.size[0] // 2
    print(n)
    plt.plot(v_R.data[n,n])
    plt.plot(nt_R.data[n,n])
    plt.show()
    charges = [(0.9, rc1, 0.0),
               (-0.9, 0.3, 0.0),
               (0.7, rc2, d12),
               (-0.7, 0.4, d12)]
    e0 = 0.0
    for q1, rc1, p1 in charges:
        for q2, rc2, p2 in charges:
            d = abs(p1 - p2)
            e0 += q1 * q2 * c(d, rc1, rc2)
    print(e0)
    print(0.9**2 / d12)


if __name__ == '__main__':
    rc1 = 0.1
    rc2 = 0.2
    print(c(8.0, rc1, rc2) / (4 * np.pi)**2)
    test_psolve()
