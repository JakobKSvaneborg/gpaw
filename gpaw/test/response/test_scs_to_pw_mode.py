import numpy as np
import pytest
from gpaw.response.df import DielectricFunction


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
    _, eps_w = calc_df(gpw_files["si_pw"])
    _, eps_scs_w = calc_df(gpw_files["si_scs_gpw"])

    # we use a relatively high tolerance since we wouldn't expect agreement
    # given the unconverged basis sets
    assert eps_scs_w.real == pytest.approx(eps_w.real, rel=0.1)
