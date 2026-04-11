import numpy as np
import matplotlib.pyplot as plt
from gpaw.new.ase_interface import GPAW

# --- Band structure calculation ---
gs_gpw = 'gs_scs.gpw'
calc = GPAW(gs_gpw).fixed_density(
    nbands=60,
    symmetry='off',
    kpts={'path': 'GKMG', 'npoints': 200},
    convergence={'bands': 30},
)
calc.write('bs.gpw', mode='all')  # mode='all' saves LCAO coefficients

# --- Get band structure object (energies, path, reference) ---
bs = calc.band_structure()
energies = bs.energies  # (nspins, nkpts, nbands)
nspins, nkpts, nbands = energies.shape

# --- Compute layer projections from LCAO coefficients ---
ibzwfs = calc.dft.ibzwfs
setups = ibzwfs._wfs_u[0].setups
nao = setups.nao

# Build boolean masks for basis functions on each layer
tags = calc.atoms.get_tags()
layer0_mask = np.zeros(nao, dtype=bool)
layer1_mask = np.zeros(nao, dtype=bool)
for a, tag in enumerate(tags):
    M = setups.M_a[a]
    M_slice = slice(M, M + setups[a].nao)
    if tag == 0:
        layer0_mask[M_slice] = True
    else:
        layer1_mask[M_slice] = True

# Compute Mulliken weights for each (spin, kpt, band)
weights_layer0 = np.zeros((nspins, nkpts, nbands))
weights_layer1 = np.zeros((nspins, nkpts, nbands))

for wfs in ibzwfs:
    s, k = wfs.spin, wfs.k
    C_nM = wfs.C_nM.gather(broadcast=True).data  # (nbands, nao)
    S_MM = wfs.S_MM.gather(broadcast=True).data   # (nao, nao)

    # Mulliken decomposition: w_nM = Re((C @ S) * C*)
    CS_nM = C_nM @ S_MM
    w_nM = np.real(CS_nM * C_nM.conj())

    weights_layer0[s, k] = w_nM[:, layer0_mask].sum(axis=1)
    weights_layer1[s, k] = w_nM[:, layer1_mask].sum(axis=1)

# Sanity check: Mulliken weights should sum to ~1
total = weights_layer0 + weights_layer1
print(f'Mulliken weight sum: min={total.min():.6f}, max={total.max():.6f}')

# --- Set up matplotlib figure (replicating bs.plot() layout) ---
fig, ax = plt.subplots()

# K-path x-coordinates and high-symmetry labels
xcoords, label_xcoords, orig_labels = bs.get_labels()
label_xcoords = list(label_xcoords)


def pretty(kpt):
    if kpt == 'G':
        return r'$\Gamma$'
    elif len(kpt) == 2:
        return kpt[0] + '$_' + kpt[1] + '$'
    return kpt


labels = [pretty(name) for name in orig_labels]

# Merge adjacent special points at the same x-coordinate
i = 1
while i < len(labels):
    if label_xcoords[i - 1] == label_xcoords[i]:
        labels[i - 1] = labels[i - 1] + ',' + labels[i]
        labels.pop(i)
        label_xcoords.pop(i)
    else:
        i += 1

# Vertical lines at special k-points (skip endpoints)
for x in label_xcoords[1:-1]:
    ax.axvline(x, color='0.5')

# Fermi level / reference energy
ax.axhline(bs.reference, color='k', ls=':')

# Axis labels and limits
ax.set_xticks(label_xcoords)
ax.set_xticklabels(labels)
ax.set_ylabel('Energy [eV]')
ax.axis(xmin=0, xmax=xcoords[-1], ymin=-7, ymax=-3)

# --- Scatter plot colored by layer 0 weight ---
for s in range(nspins):
    X = np.tile(xcoords, (nbands, 1)).T       # (nkpts, nbands)
    E = energies[s]                            # (nkpts, nbands)
    W = weights_layer0[s]                      # (nkpts, nbands)

    sc = ax.scatter(X.ravel(), E.ravel(), c=W.ravel(),
                    cmap='coolwarm', vmin=0, vmax=1, s=1)

fig.colorbar(sc, ax=ax, label='Layer 0 weight')
fig.savefig('bandstructure.png', dpi=200)
print('Saved bandstructure.png')
