from gpaw.new.pw.hybridsk import PWHybridHamiltonianK


def test_apply3():
    ham = PWHybridHamiltonianK(
        grid: UGDesc,
        pw: PWDesc,
        xc,
        setups: Setups,
        relpos_ac,
        atomdist,
        log,
        kpt_comm,
        band_comm,
        comm)
    ...
