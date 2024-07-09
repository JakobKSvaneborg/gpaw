from gpaw.lrtddft import LrTDDFT


def test_rpa(H2):
    lr = LrTDDFT(H2.calc, xc='RPA', restrict={'to': [1]})
    assert len(lr) == 1
