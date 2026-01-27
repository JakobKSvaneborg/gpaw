import numpy as np
from gpaw.response.bse import BSE


def test_si_scs(in_tmp_dir, gpw_files):

    bse = BSE(
        calc=gpw_files['si_scs_gpw'],
        ecut=30,
        valence_bands=3,
        conduction_bands=2,
        mode="BSE",
    )

    bse.get_polarizability(
        eta=0.1, w_w=np.linspace(0, 50, 101), filename="test_si_scs.csv"
    )
