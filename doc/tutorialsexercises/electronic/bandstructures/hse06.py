import pickle
from pathlib import Path

from ase.build import mx2
from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW
from gpaw.new.pw.nschse import NonSelfConsistentHSE06


def mos2():
    """Do LDA calculation for MoS2 layer."""
    atoms = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.13,
                size=(1, 1, 1))
    atoms.center(vacuum=3.5, axis=2)
    k = 6
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 400},
                      kpts=(k, k, 1),
                      txt='lda.txt')
    atoms.get_potential_energy()
    return atoms


def bandstructure(gs_calc, bp):
    """Calculate HSE06 bandstructure on top of LDA."""
    fermi_level = gs_calc.get_fermi_level()
    vacuum_level = gs_calc.dft.vacuum_level()
    N = 13 + 4  # 13 occupied + 4 empty
    bs_calc = gs_calc.fixed_density(
        kpts=bp,
        convergence={'bands': N},
        symmetry='off',
        txt='gmkg.txt')
    lda_skn = bs_calc.eigenvalues()
    hse = NonSelfConsistentHSE06.from_dft_calculation(
        gs_calc.dft, log='hse06.txt')
    hse_skn = hse.calculate(bs_calc.dft.ibzwfs, na=0, nb=N)
    # Return energies relative to vacuum level:
    return (lda_skn[0, :, :N] - vacuum_level,
            hse_skn[0] - vacuum_level,
            fermi_level - vacuum_level)


def run():
    atoms = mos2()
    bp = atoms.cell.bandpath('GMKG', npoints=50)
    lda_kn, hse_kn, fermi_level = bandstructure(atoms.calc, bp)
    if world.rank == 0:
        Path('bs.pckl').write_bytes(
            pickle.dumps((bp, lda_kn, hse_kn, fermi_level)))


if __name__ == '__main__':
    run()
