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
                      txt='lda.txt')
    atoms.get_potential_energy()
    return atoms


def bandstructure(calc, bp):
    hse = NonSelfConsistentHSE06.from_dft_calculation(calc.dft)
    fermi_level = calc.get_fermi_level()
    vacuum_level = calc.dft.vacuum_level()
    N = 13 + 4
    bs_calc = calc.fixed_density(
        kpts=bp,
        convergence={'bands': N},
        symmetry='off')
    lda_skn = bs_calc.eigenvalues()
    hse = NonSelfConsistentHSE06.from_dft_calculation(calc.dft)
    hse_skn = hse.calculate(bs_calc.dft.ibzwfs)
    return (lda_skn[0, :, :N] - vacuum_level,
            hse_skn[0, :, :N] - vacuum_level,
            fermi_level - vacuum_level)


def run():
    atoms = mos2()
    bp = atoms.cell.bandpath('GMKG', npoints=50)
    lda_kn, hse_kn, fermi_level = bandstructure(atoms.calc, bp)
    if world.rank == 0:
        Path('bs.pckl').write_bytes(
            pickle.dumps((bp, lda_kn, hse_kn, fermi_level)))


def plot(bp, lda_kn, hse_kn, fermi_level):
    ax = plt.subplot()
    x, xlabels, labels = bp.get_linear_kpoint_axis()
    labels = [label.replace('G', r'$\Gamma$') for label in labels]
    for y in lda_kn.T:
        ax.plot(x, y, color='C0')
    for y in hse_kn.T:
        ax.plot(x, y, color='C1')
    ax.hlines(fermi_level, 0.0, x[-1])
    ax.set_xlim(0.0, x[-1])
    ax.set_ylim(fermi_level - 3.0, fermi_level + 3.0)
    ax.set_xticks(xlabels)
    ax.set_xticklabels(labels)
    plt.show()
    # plt.savefig('hse.png')


if __name__ == '__main__':
    path = Path('bs.pckl')
    if not path.is_file():
        run()
    else:
        plot(*pickle.loads(path.read_bytes()))
