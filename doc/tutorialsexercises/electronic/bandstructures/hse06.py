from ase.build import mx2
from gpaw.new.ase_interface import GPAW
import matplotlib.pyplot as plt
import numpy as np
from ase.units import Ha
from gpaw.new.ase_interface import GPAW


def mos2() -> None:
    """MoS2 layer."""
    atoms = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.13,
                size=(1, 1, 1))
    atoms.center(vacuum=3.5, axis=2)
    k = 6
    atoms.calc = GPAW(mode={'nama': 'pw', 'ecut': 500},
                      kpts=(k, k, 1),
                      txt='lda.txt')
    atoms.get_potential_energy()

    bp = atoms.cell.bandpath('GMKG', npoints=50)
    bs_calc = atoms.calc.fixed_density(kpts=bp, symmetry='off')


def plot(ibzwfs, bp, ax):
    x_k, xlabel_K, label_K = bp.get_linear_kpoint_axis()
    label_K = [label.replace('G', r'$\Gamma$') for label in label_K]
    ax.set_xlim(0, x_k[-1])
    ax.set_ylim(-10, 0)
    ax.set_xticks(xlabel_K)
    ax.set_xticklabels(label_K)
    return lines


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 3, sharey=True)
    for i, ax in enumerate(axs):
        bs_calc = GPAW(f'mos2ws2-{i}.gpw')
        bp = bs_calc.atoms.cell.bandpath('GMKG', npoints=50)
        lines = plot(bs_calc.dft.ibzwfs, bp, ax)
    # plt.show()
    plt.savefig('mos2ws2.png')
