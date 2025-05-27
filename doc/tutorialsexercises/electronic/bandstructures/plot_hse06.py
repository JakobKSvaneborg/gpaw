# wep-page: hse06.png
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


def plot(bp, lda_kn, hse_kn, fermi_level):
    ax = plt.subplot()
    x, xlabels, labels = bp.get_linear_kpoint_axis()
    labels = [label.replace('G', r'$\Gamma$') for label in labels]
    label = 'LDA'
    for y in lda_kn.T:
        ax.plot(x, y, color='C0', label=label)
        label = None
    label = 'HSE06@LDA'
    for y in hse_kn.T:
        ax.plot(x, y, color='C1', label=label)
        label = None
    ax.hlines(fermi_level, 0.0, x[-1], colors='black', label='Fermi-level')
    ax.legend()
    ax.set_xlim(0.0, x[-1])
    ax.set_ylim(fermi_level - 3.0, fermi_level + 3.0)
    ax.set_xticks(xlabels)
    ax.set_xticklabels(labels)
    ax.set_ylabel('eigenvalues relative to vacuum-level [eV]')
    # plt.show()
    plt.savefig('hse06.png')


if __name__ == '__main__':
    path = Path('bs.pckl')
    plot(*pickle.loads(path.read_bytes()))
