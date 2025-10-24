import pytest
import numpy as np
from gpaw.response.bse import BSE


@pytest.mark.response
def test_response_bse_parse_bands(in_tmp_dir, gpw_files):

    bse = BSE(gpw_files['mos2_pw'],
              ecut=10,
              valence_bands=4,
              conduction_bands=3,
              eshift=0.8,
              nbands=15)

    # Check consistency with written results
    n_valence_bands = int(bse.gs.nvalence / 2)
    correct_valence_n = range(n_valence_bands - 4, n_valence_bands)
    correct_conduction_n = range(n_valence_bands, n_valence_bands + 3)

    assert np.array_equal(correct_valence_n, bse.val_m)
    assert np.array_equal(correct_conduction_n, bse.con_m)

    bse = BSE(gpw_files['mos2_pw'],
              ecut=10,
              add_soc=True,
              valence_bands=8,
              conduction_bands=6,
              eshift=0.8,
              nbands=15)

    # Check consistency with written results
    n_valence_bands = bse.gs.nvalence
    correct_valence_n = range(n_valence_bands - 8, n_valence_bands)
    correct_conduction_n = range(n_valence_bands, n_valence_bands + 6)

    assert np.array_equal(correct_valence_n, bse.val_m)
    assert np.array_equal(correct_conduction_n, bse.con_m)
    gs = bse.gs

    with pytest.raises(ValueError,
                       match='The bands must be specified as a single *'):
        BSE.parse_bands(bands=[range(4), range(4)], band_type='valence',
                        gs=gs)

    with pytest.raises(ValueError,
                       match='10000 valence bands were requested *'):
        BSE.parse_bands(bands=10000, band_type='valence', gs=gs)

    with pytest.raises(ValueError,
                       match='\'bands\' must be a *'):
        BSE.parse_bands(bands=-1, band_type='valence', gs=gs)
