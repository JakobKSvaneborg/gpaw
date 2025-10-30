from gpaw.doctools.makebadge import main


def test_coverage_badge(capsys):
    main(['(ignored)', '42'])
    svg, _ = capsys.readouterr()
    assert 'Coverage' in svg
    assert '42%' in svg
