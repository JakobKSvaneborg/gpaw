import pytest
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE
import numpy as np


@pytest.mark.response
def test_bse_exclude_states():  # in_tmp_dir, gpw_files):
    a = 5.431  # From PRB 73,045112 (2006)
    atoms = bulk('Si', 'diamond', a=a)
    atoms.positions -= a / 8
    eshift = 0.8
    calc = GPAW(mode='pw',
                kpts={'size': (2, 2, 2), 'gamma': True},
                occupations=FermiDirac(0.001),
                nbands=12,
                convergence={'bands': -4})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si.gpw', 'all')
    bse = BSE('Si.gpw',
              ecut=50.,
              valence_bands=range(1, 4),
              conduction_bands=range(4, 6),
              deps_max=6,
              eshift=eshift,
              nbands=8)
    bse_matrix = bse.calculate(optical=True)
    excludedeps_S = np.where(bse_matrix.deps_S > bse_matrix.deps_max)[0]
    excludef_S = np.where(np.abs(bse_matrix.df_S) < 0.001)[0]
    exclude_S = np.unique(np.concatenate((excludef_S, excludedeps_S)))
    w_T, v_Rt, exclude_S = bse_matrix.diagonalize_tammdancoff(bse)
    nk = (calc.wfs.kd.nbzkpts)
    nval = 3
    ncond = 2
    n_pairs = nk * nval * ncond
    assert len(exclude_S) == 14
    assert len(w_T) == n_pairs - len(exclude_S)
    print(w_T)
    assert w_T[0] == pytest.approx(0.1004, abs=0.001)
    assert w_T[11] == pytest.approx(0.1287, abs=0.001)
    assert w_T[29] == pytest.approx(0.1924, abs=0.001)