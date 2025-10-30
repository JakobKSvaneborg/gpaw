from gpaw.doctools.makebadge import coverage_badge


def test_coverage_badge():
    svg = coverage_badge(42)
    assert 'Coverage' in svg
    assert '42%' in svg
