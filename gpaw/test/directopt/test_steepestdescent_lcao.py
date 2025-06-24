import pytest

from gpaw import GPAW


@pytest.mark.do
def test_steepestdescent_lcao(in_tmp_dir, gpw_files):
    """
    Test steepest descent and conjugate gradients
    search direction algorithms
    :param in_tmp_dir:
    :return:
    """

    calc = GPAW(gpw_files['h3_do_sd_lcao'])
    atoms = calc.atoms
    atoms.calc = calc

    for sd_algo in ['sd', 'fr-cg']:
        calc = calc.new(eigensolver={'name': 'etdm-lcao',
                                     'searchdir_algo': sd_algo})
        atoms.calc = calc
        e = atoms.get_potential_energy()
        assert e == pytest.approx(6.021948, abs=1.0e-5)
