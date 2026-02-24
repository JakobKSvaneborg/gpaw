import numpy as np
import pytest
from gpaw.response.df import DielectricFunction


def calc_df(gpwfile, world=None):
    kwargs = {}
    if world is not None:
        kwargs['world'] = world
    df = DielectricFunction(
        calc=gpwfile,
        ecut=30,
        nbands=6,
        eta=0.1,
        hilbert=False,
        frequencies=np.linspace(0, 1, 10),
        **kwargs)
    return df.get_dielectric_function()


def test_si_scs(in_tmp_dir, gpw_files, gpaw_new, mpi):
    if not gpaw_new:
        pytest.skip()
    _, eps_w = calc_df(gpw_files['si_pw'], world=mpi.comm)
    dft = mpi.GPAW(gpw_files['si_scs_lcao']).dft
    dft.change(eigensolver={})  # remove SCS solver which PW-mode doesn't like
    dft.change_mode('pw')
    dft.ase_calculator().write('si_scs_pw.gpw', 'all')
    _, eps_scs_w = calc_df('si_scs_pw.gpw', world=mpi.comm)

    # we use a relatively high tolerance since we wouldn't expect agreement
    # given the unconverged basis sets
    assert eps_scs_w.real == pytest.approx(eps_w.real, rel=0.1)
