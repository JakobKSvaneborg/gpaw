"""Submit tests from the test suite that were removed because they were
too long."""

from myqueue.workflow import run

from gpaw import GPAW_NEW


def workflow():
    run(script='H2Al110.py')
    if not GPAW_NEW:
        run(script='dscf_CO.py')
    run(script='revtpss_tpss_scf.py')
    run(script='ltt.py')
    run(script='scalapack.py', cores=16)
    run(script='pblacs_oblong.py', cores=48)
