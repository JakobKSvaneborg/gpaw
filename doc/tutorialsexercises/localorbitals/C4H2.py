# creates: C4H2_minimal.png, C4H2_extended.png
from ase.build import graphene_nanoribbon
from gpaw import GPAW
from gpaw.lcao.local_orbitals import LocalOrbitals
from gpaw.lcao.tightbinding import TightBinding
from matplotlib import pyplot as plt

plt.style.use('bmh')
plt.rcParams['font.size'] = 13
plt.rcParams['lines.linewidth'] = 2


def compare_bandstructure(lcao, los, figname, figtitle):
    """Compare bands between LCAO and LO model.

    Parameters
    ----------
    lcao : TightBinding
        A TB wrapper aroubd an LCAO calculation.
    los : LocalOrbitals
        A LO wrapper around an LCAO calculation.
    figname : str
        Save the figure using `figname`.
    figtitle : str
        Title of the figure
    """
    # Define a bandpath
    bp = calc.atoms.cell.bandpath('GX', npoints=60)
    xcoords, special_xcoords, labels = bp.get_linear_kpoint_axis()
    # Get the bandstructure
    lcao_eps = lcao.band_structure(bp.kpts, blochstates=False)
    los_eps = los.band_structure(bp.kpts, blochstates=False)
    # Plot
    plt.figure()
    fermi = lcao.calc.get_fermi_level()
    lines = plt.plot(xcoords, lcao_eps - fermi, 'tab:blue')
    lines[0].set_label('LCAO')
    lines = plt.plot(xcoords, los_eps - fermi,
                     '-.', color='tab:orange')
    lines[0].set_label('LOs')
    plt.legend()
    plt.hlines(0.,
               xmin=special_xcoords[0], xmax=special_xcoords[1],
               color='k', linestyle='--')
    plt.ylim(-10., 10.)
    plt.title(figtitle)
    plt.xticks(special_xcoords, labels)
    plt.ylabel('Energy (eV)')
    plt.savefig(figname, bbox_inches='tight')


# Atoms
gnr = graphene_nanoribbon(2, 1, type='zigzag', saturated=True,
                          C_H=1.1, C_C=1.4, vacuum=5.0)

# LCAO calculation
calc = GPAW(legacy_gpaw=True,
            mode='lcao',
            xc='LDA',
            basis='szp(dzp)',
            txt=None,
            kpts={'size': (1, 1, 11),
                  'gamma': True},
            symmetry={'point_group': False,
                      'time_reversal': True})
gnr.calc = calc
gnr.get_potential_energy()

# Start post-process
tb = TightBinding(calc.atoms, calc)

# Construct a LocalOrbital Object
los = LocalOrbitals(calc)

# Subdiagonalize carbon atoms and group by symmetry and energy.
los.subdiagonalize('C', groupby='symmetry')

# Take minimal model
los.take_model(minimal=True)

# Compare the bandstructure of the effective model and compare it with LCAO
compare_bandstructure(tb, los, "C4H2_minimal.png", "minimal=True")

# Extend with groups of LOs that overlap with the minimal model
los.take_model(minimal=False)

# Compare the bandstructure of the effective model and compare it with LCAO
compare_bandstructure(tb, los, "C4H2_extended.png", "minimal=False")
