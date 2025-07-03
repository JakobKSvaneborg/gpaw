from ase import Atoms
from gpaw import GPAW, PW


def exx(n, xc):
    L = 2.6
    a = Atoms('H',
              cell=[L, L, 1],
              pbc=1)
    a *= (1, 1, n)
    a.calc = GPAW(
        mode=PW(400, force_complex_dtype=0),
        eigensolver={'name': 'davidson', 'niter': 1},
        # eigensolver={'name': 'rmm-diis'},
        symmetry='off',
        # setups='ae',
        # kpts={'size': (1, 1, 4 // n), 'gamma': True},
        convergence={'density': 1e-6},
        txt=f'n{n}.txt',
        xc=xc)
    e = a.get_potential_energy()
    print(e / n)
    eps1 = a.calc.get_eigenvalues(0)
    print(eps1)
    if n == 1:
        eps2 = a.calc.get_eigenvalues(1)
        print(eps2)


if __name__ == '__main__':
    import sys
    exx(int(sys.argv[1]), sys.argv[2])
