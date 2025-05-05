# creates: O_nm.png, P_nm_proj.png
import numpy as np


def simple_swap(inseq, ii, jj):

    seq = inseq.copy()
    # just swap two elements
    buf = seq[ii].copy()
    seq[ii] = seq[jj].copy()
    seq[jj] = buf

    return seq


def get_overlaps():

    np.random.seed(12347)
    #                   0   1   2   3   4   5   6   7
    numbers = np.array([1., 1., 1., 1., 0., .4, .6, 0.])
    nbands = 8
    noise = 0.15

    # generate overlap
    perm = np.arange(nbands)
    perm = simple_swap(perm, 1, 2)
    perm = simple_swap(perm, 4, 6)
    Xi = np.random.normal(scale=noise, size=(nbands, nbands))
    O_nm = np.eye(nbands)[:, perm] + Xi

    # properly normalize
    O_nm = np.abs(O_nm)
    O_nm /= O_nm.max()

    # updated occupations
    # here: manually
    f_n = simple_swap(numbers, 4, 6)

    return nbands, numbers, f_n, O_nm


def get_projections(numbers, O_nm, proj=True):

    # calculate subspace projections
    mask_1 = numbers == 1.
    f_s = np.r_[numbers[mask_1], .0, .4, .6]

    P_nm = np.zeros_like(O_nm)
    if proj:
        P_nm[mask_1, :] = (np.sum(O_nm[mask_1, :]**2, axis=0)**0.5)[None, :]
    else:
        P_nm[mask_1, :] = np.max(O_nm[mask_1, :], axis=0)[None, :]

    # for subspaces with only one orbital
    # we do the assignment manually here
    P_nm[5, 5] = O_nm[5, 5]
    P_nm[6, 4] = O_nm[4, 6]

    return f_s, P_nm


def plot_overlaps_projections(proj=True):
    from matplotlib import pyplot as plt

    nbands, numbers, f_n, O_nm = get_overlaps()
    f_s, P_nm = get_projections(numbers, O_nm, proj=proj)

    # overlaps
    fig1 = plt.figure(figsize=(5, 5), constrained_layout=True)
    gs = fig1.add_gridspec(5, 5)
    ax1 = fig1.add_subplot(gs[1:5, 0:4])
    ax2 = fig1.add_subplot(gs[0, 0:4], sharex=ax1)
    ax3 = fig1.add_subplot(gs[1:5, 4], sharey=ax1)

    # matrix
    ax1.imshow(O_nm.T, aspect='auto', cmap='Greys')
    ax1.set_xlabel('$|O_{nm}^{(k)}|$')

    # initial occupations
    ax2.bar(np.arange(nbands), numbers.tolist(), color='darkblue')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('$f_n^0$', rotation=0)

    # final occupations
    ax3.barh(np.arange(nbands), f_n, color='r')
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_xlabel('$f_m^{(k)}$')

    # projections
    fig2 = plt.figure(figsize=(5, 5), constrained_layout=True)
    gs = fig2.add_gridspec(5, 5)
    ax4 = fig2.add_subplot(gs[1:5, 0:4])
    ax5 = fig2.add_subplot(gs[0, 0:4], sharex=ax4)
    ax6 = fig2.add_subplot(gs[1:5, 4], sharey=ax4)

    # matrix
    ax4.imshow(P_nm.T, aspect='auto', cmap='Greys')
    if proj:
        ax4.set_xlabel('$P_{sm}^{(k)}$ (projections)')
    else:
        ax4.set_xlabel('$P_{sm}^{(k)}$ (maximum)')

    # subspace occupations (number corresponding to subspace size)
    ax5.bar(np.arange(len(f_s)), f_s, color='darkblue')
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.set_ylabel('$f_n^s$', rotation=0)

    # final occupations
    ax6.barh(np.arange(nbands), f_n, color='r')
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    ax6.set_xlabel('$f_m^{(k)}$')

    fig1.savefig('O_nm.png')
    if proj:
        fig2.savefig('P_nm_proj.png')
    else:
        fig2.savefig('P_nm_max.png')


plot_overlaps_projections()
