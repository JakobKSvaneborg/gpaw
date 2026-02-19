import numpy as np
import pytest
from gpaw.response.df import DielectricFunction
from gpaw.dft import GPAW


def calc_df(gpwfile):
    df = DielectricFunction(
        calc=gpwfile,
        ecut=30,
        nbands=6,
        eta=0.1,
        hilbert=False,
        frequencies=np.linspace(0, 1, 10),
    )
    return df.get_dielectric_function()


def test_si_scs(in_tmp_dir, gpw_files):
    _, eps_w = calc_df(gpw_files['si_pw'])
    pw = GPAW(gpw_files['si_scs_lcao']).dft.change_mode('pw')
    pw.ase_calculator().write('tmp.gpw', 'all')
    _, eps_scs_w = calc_df('tmp.gpw')

    # we use a relatively high tolerance since we wouldn't expect agreement
    # given the unconverged basis sets
    assert eps_scs_w.real == pytest.approx(eps_w.real, rel=0.1)
