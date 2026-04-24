import pytest

from gpaw import GPAW
from gpaw.mpi import world


@pytest.mark.do
def test_etdm_maxiter_with_localize_every(in_tmp_dir, gpw_files):
    """Test that maxiter counts only outer SCF steps, not inner PZ steps.

    Verifies fix for TheochemUI/gpaw#12: when localize_every is set, the
    periodic PZ-SIC inner loop uses subspace_iters (not iters), so inner
    loop steps do not count against maxiter.
    """
    if world.size == 8:
        pytest.skip('See #1406')

    from gpaw import restart
    calc = GPAW(gpw_files['h2o_cdo_lcao_sic'])
    calc.write('h2o_restart.gpw', mode='all')
    H2O, calc = restart('h2o_restart.gpw', txt='-')

    maxiter = 3
    H2O.calc.results.pop('energy')
    H2O.calc.scf.converged = False
    calc.set(eigensolver={'name': 'etdm-lcao',
                          'localizationtype': 'pz',
                          'localize_every': 1,
                          'subspace_convergence': 1e-3,
                          'functional': {'name': 'PZ-SIC',
                                         'scaling_factor': (0.5, 0.5)},
                          'need_init_orbs': False},
             convergence={'eigenstates': 1e-4})
    calc.parameters['maxiter'] = maxiter

    H2O.get_potential_energy()
    niter = calc.get_number_of_iterations()

    # niter must not exceed maxiter; inner PZ steps must not inflate it
    assert niter <= maxiter, (
        f'niter={niter} exceeded maxiter={maxiter}; '
        'inner PZ localization steps may be counted as outer iterations')
