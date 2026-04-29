import numpy as np
from ase.build import mx2
from gpaw import GPAW, MixerFull
from gpaw.occupations import FermiDirac
from gpaw.spinorbit import soc_eigenstates
from gpaw.occupations import create_occ_calc


def check_ani(Ez, Ex, target):
    print(Ez - Ex)
    assert abs(Ez - Ex - target) < 0.000020


params = dict(mode={'name': 'pw', 'ecut': 800},
              kpts={'size': (12, 12, 1),
                    'gamma': True},
              mixer=MixerFull())
occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': 0.001})

"""Self-consistent SOC."""
a = mx2('NiBr2', kind='1T', a=3.7)
a.center(vacuum=3, axis=2)

magmoms = np.zeros((3, 3))
magmoms[0, 0] = 2
a.calc = GPAW(magmoms=magmoms,
              soc=True,
              symmetry='off',
              occupations=FermiDirac(width=0.001),
              parallel={'domain': 1, 'band': 1},
              **params)
Ex = a.get_potential_energy()

magmoms = np.zeros((3, 3))
magmoms[0, 2] = 2
a.calc = GPAW(magmoms=magmoms,
              soc=True,
              symmetry='off',
              occupations=FermiDirac(width=0.001),
              parallel={'domain': 1, 'band': 1},
              **params)
Ez = a.get_potential_energy()
check_ani(Ez, Ex, 0.000185)


"""Non-collinear plus SOC."""
a.calc = GPAW(magmoms=magmoms,
              soc=False,
              convergence={'bands': 38},
              symmetry='off',
              parallel={'domain': 1, 'band': 1},
              **params)
a.get_potential_energy()

bzwfs = soc_eigenstates(a.calc, n2=38, occcalc=occcalc)
Ez = bzwfs.calculate_band_energy()
bzwfs = soc_eigenstates(a.calc, n2=38, theta=90, occcalc=occcalc)
Ex = bzwfs.calculate_band_energy()
check_ani(Ez, Ex, 0.000270)


"""Collinear plus SOC."""
a.set_initial_magnetic_moments([2, 0, 0])
a.calc = GPAW(convergence={'bands': 19},
              **params)
a.get_potential_energy()

bzwfs = soc_eigenstates(a.calc, n2=19, occcalc=occcalc)
Ez = bzwfs.calculate_band_energy()
bzwfs = soc_eigenstates(a.calc, n2=19, theta=90, occcalc=occcalc)
Ex = bzwfs.calculate_band_energy()
check_ani(Ez, Ex, 0.000247)

# Ideally the last two checks should yield the same value.
# But the bands are included in slightly different ways for
# collinear and non-collinear calculations in the non-
# selfconsistent SOC so there are small differences.
