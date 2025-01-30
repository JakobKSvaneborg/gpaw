import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.build import mx2
from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW
from gpaw.new.pw.nschse import NonSelfConsistentHSE06


def mos2() -> None:
    """MoS2 layer."""
    atoms = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.13,
                size=(1, 1, 1))
    atoms.center(vacuum=3.5, axis=2)
    k = 3
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 400},
                      kpts=(k, k, 1),
                      # symmetry='off',
                      txt='lda.txt')
    atoms.get_potential_energy()
    return atoms


def bandstructure(calc, bp):
    from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
    hse = NonSelfConsistentHSE06.from_dft_calculation(calc.dft)
    if 1:
        e0, v0, v = non_self_consistent_eigenvalues(calc, 'HSE06')
        hse_n = hse.calculate(calc.dft.ibzwfs)
        print(hse_n - (e0 - v0 + v))
        1 / 0
    efermi = calc.get_fermi_level()
    bs_calc = calc.fixed_density(kpts=bp, symmetry='off')
    lda_skn = bs_calc.eigenvalues()
    print(lda_skn)
    hse = NonSelfConsistentHSE06.from_dft_calculation(calc.dft)
    hse_skn = hse.calculate(bs_calc.dft.ibzwfs)
    print(hse_skn)
    return lda_skn[0] - efermi, hse_skn[0] - efermi


def run():
    atoms = mos2()
    bp = atoms.cell.bandpath('GMKG', npoints=50)
    # bp = atoms.cell.bandpath('KG', npoints=10)
    lda_kn, hse_kn = bandstructure(atoms.calc, bp)
    if world.rank == 0:
        Path('bs.pckl').write_bytes(pickle.dumps((bp, lda_kn, hse_kn)))


def plot(bp, lda_kn, hse_kn):
    ax = plt.subplot()
    x, xlabels, labels = bp.get_linear_kpoint_axis()
    labels = [label.replace('G', r'$\Gamma$') for label in labels]
    for y in lda_kn.T:
        ax.plot(x, y, color='C0')
    for y in hse_kn.T:
        ax.plot(x, y, color='C1')
    ax.set_xlim(0.0, x[-1])
    ax.set_ylim(-3.0, 3.0)
    ax.set_xticks(xlabels)
    ax.set_xticklabels(labels)
    plt.show()
    # plt.savefig('hse.png')


if __name__ == '__main__':
    path = Path('bs?.pckl')
    if not path.is_file():
        run()
    else:
        plot(*pickle.loads(path.read_bytes()))
