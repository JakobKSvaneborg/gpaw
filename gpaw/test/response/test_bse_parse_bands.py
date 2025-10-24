import pytest
import numpy as np
from gpaw.response.bse import BSE
from gpaw.response.groundstate import ResponseGroundStateAdapter


@pytest.mark.response
def test_response_bse_parse_bands(in_tmp_dir, gpw_files):
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files['mos2_pw'])
    val_m = BSE.parse_bands(bands=4, gs=gs, band_type='valence', add_soc=False)
    con_m = BSE.parse_bands(bands=3, gs=gs, band_type='conduction',
                            add_soc=False)

    # Check consistency with written results
    n_valence_bands = int(gs.nvalence / 2)
    correct_valence_n = range(n_valence_bands - 4, n_valence_bands)
    correct_conduction_n = range(n_valence_bands, n_valence_bands + 3)

    assert np.array_equal(correct_valence_n, val_m)
    assert np.array_equal(correct_conduction_n, con_m)

    bse = BSE(gpw_files['mos2_pw'],
              ecut=10,
              add_soc=True,
              valence_bands=8,
              conduction_bands=6,
              eshift=0.8,
              nbands=15)

    # Check consistency with written results
    val_m = BSE.parse_bands(bands=8, gs=gs, band_type='valence', add_soc=True)
    con_m = BSE.parse_bands(bands=6, gs=gs, band_type='conduction',
                            add_soc=True)

    n_valence_bands = gs.nvalence
    correct_valence_n = range(n_valence_bands - 8, n_valence_bands)
    correct_conduction_n = range(n_valence_bands, n_valence_bands + 6)

    assert np.array_equal(correct_valence_n, val_m)
    assert np.array_equal(correct_conduction_n, con_m)
    gs = bse.gs

    with pytest.raises(ValueError,
                       match='The bands must be specified as a single *'):
        BSE.parse_bands(bands=[range(4), range(4)], band_type='valence',
                        gs=gs, add_soc=False)

    with pytest.raises(ValueError,
                       match='10000 valence bands were requested *'):
        BSE.parse_bands(bands=10000, band_type='valence', gs=gs, add_soc=False)

    with pytest.raises(ValueError,
                       match='\'bands\' must be a *'):
        BSE.parse_bands(bands=-1, band_type='valence', gs=gs, add_soc=False)
