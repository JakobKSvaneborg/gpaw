# creates: C6H6_minimal.png, C6H6_extended.png, C6H6_pzLOs.png
import numpy as np
from ase.build import molecule
from gpaw import GPAW
from gpaw.lcao.local_orbitals import LocalOrbitals
from gpaw.lcao.pwf2 import LCAOwrap
from matplotlib import pyplot as plt
from scipy.linalg import eigvalsh

plt.style.use('bmh')
plt.rcParams['font.size'] = 13
plt.rcParams['lines.linewidth'] = 2


def get_eigvals(model, erng=[-10, 10]):
    """Helper function to get the eigenvalues.

    Parameters
    ----------
    model : Object
        Must have a `get_hamiltonian` and `get_overlap` methods.
    erng : (float,float), optional
        Energy range of min and max eigenvalues.
    """
    # Compute eigenvalues from H and S.
    H = model.get_hamiltonian()
    S = model.get_overlap()
    if np.allclose(S, np.eye(S.shape[0]), atol=1e-4):
        eigs = np.linalg.eigvalsh(H)
    else:
        eigs = eigvalsh(H, S)
    eigs = eigs[(eigs > erng[0]) & (eigs < erng[1])]
    return eigs


def compare_eigvals(lcao, los, figname, figtitle):
    """Compare eigenvalues between LCAO and LO model.

    Parameters
    ----------
    lcao : LCAOWrap
        An LCAO wrapper around an LCAO calculation.
    los : LocalOrbitals
        A LO wrapper around an LCAO calculation
    figname : str
        Save the figure using `figname`.
    figtitle : str
        Title of the figure
    """
    # Get eigenvalues
    fermi = lcao.calc.get_fermi_level()
    erng = [fermi + elim for elim in [-10, 10]]
    lcao_eigs = get_eigvals(lcao, erng)
    los_eigs = get_eigvals(los, erng)
    # Plot eigenvalues
    plt.figure()
    plt.hlines(lcao_eigs, -1, -0.01, color='tab:blue')
    plt.hlines(los_eigs, 0.01,
               1, linestyles='-.', color='tab:orange')
    plt.hlines(0., -1., 1., linestyle='--', color='black')
    plt.grid(axis='x')
    _ = plt.xticks(labels=['LCAO', 'LOs'], ticks=[-0.5, 0.5])
    plt.title(figtitle)
    plt.ylabel('Energy (eV)')
    plt.savefig(figname, bbox_inches='tight')


# Atoms
benzene = molecule('C6H6', vacuum=5)

# LCAO calculation
calc = GPAW(legacy_gpaw=True, mode='lcao', xc='LDA', basis='szp(dzp)', txt=None)
benzene.calc = calc
benzene.get_potential_energy()

# LCAO wrapper
lcao = LCAOwrap(calc)

# Construct a LocalOrbital Object
los = LocalOrbitals(calc)

# Subdiagonalize carbon atoms and group by energy.
los.subdiagonalize('C', groupby='energy')
# Groups of LOs are stored in a dictionary >>> los.groups

# Dump pz-type LOs in cube format
fig = los.plot_group(-6.9)
fig.savefig('C6H6_pzLOs.png', dpi=300, bbox_inches='tight')

# Take minimal model
los.take_model(minimal=True)
# Get the size of the model >>> len(los.model)
# Assert that the Hamiltonian has the same
# dimensions >> los.get_hamiltonian().shape

# Assert model indeed conicides with pz-type LOs
assert los.indices == los.groups[-6.9]

# Compare eigenvalues with LCAO
compare_eigvals(lcao, los, 'C6H6_minimal.png', 'minimal=True')

# Extend with groups of LOs that overlap with the minimal model
los.take_model(minimal=False)
# Get the size of the model >>> len(los.model)

# Assert model is extended with other groups.
assert los.indices == (los.groups[-6.9] + los.groups[20.2] + los.groups[21.6])

# Compare eigenvalues with LCAO
compare_eigvals(lcao, los, 'C6H6_extended.png', 'minimal=False')
