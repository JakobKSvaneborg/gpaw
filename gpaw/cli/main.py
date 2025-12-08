"""GPAW command-line tool."""
import os
import subprocess
import sys

commands = [
    ('run', 'gpaw.cli.run'),
    ('info', 'gpaw.cli.info'),
    ('test', 'gpaw.cli.test'),
    ('dos', 'gpaw.cli.dos'),
    ('gpw', 'gpaw.cli.gpw'),
    ('completion', 'gpaw.cli.completion'),
    ('atom', 'gpaw.atom.aeatom'),
    ('diag', 'gpaw.cli.fulldiag'),
    # ('quick', 'gpaw.cli.quick'),
    ('python', 'gpaw.cli.python'),
    ('sbatch', 'gpaw.cli.sbatch'),
    ('dataset', 'gpaw.atom.generator2'),
    ('plot-dataset', 'gpaw.atom.plot_dataset'),
    ('basis', 'gpaw.atom.basisfromfile'),
    ('plot-basis', 'gpaw.basis_data'),
    ('symmetry', 'gpaw.symmetry'),
    ('install-data', 'gpaw.cli.install_data')]


def hook(parser, args):
    parser.suggest_on_error = True
    parser.add_argument('-P', '--parallel', type=int, metavar='N',
                        help='Run on N CPUs.')
    args = parser.parse_args(args)

    if args.command == 'python':
        args.traceback = True

    if hasattr(args, 'dry_run'):
        N = int(args.dry_run)
        if N:
            import gpaw
            gpaw.dry_run = N
            import gpaw.mpi as mpi
            mpi.world = mpi.SerialCommunicator()
            mpi.world.size = N

    if args.parallel:
        from gpaw.mpi import compiled_with_mpi, world

        if compiled_with_mpi and world.size == 1 and args.parallel > 1:
            py = sys.executable
        elif not compiled_with_mpi:
            raise SystemExit('MPI not available')
        else:
            py = ''

        if py:
            # Start again in parallel:
            pyargs = []
            if sys.version_info >= (3, 11):
                # Don't prepend a potentially unsafe path to sys.path
                pyargs.append('-P')
            arguments = ['mpiexec',
                         *os.environ.get('GPAW_MPI_OPTIONS', '').split(),
                         '-np',
                         str(args.parallel),
                         py,
                         *pyargs,
                         '-m',
                         'gpaw',
                         *sys.argv[1:]]

            # Use a clean set of environment variables without any MPI stuff:
            p = subprocess.run(arguments, check=False,
                               env={'GPAW_MPI_INIT': '1', **os.environ})
            sys.exit(p.returncode)

    return args


def gpaw_python_init_magic():
    # We run this very early so as to set required environment variables
    # before anybody else gets to see them.
    assert 'gpaw.mpi' not in sys.modules
    assert 'ase.parallel' not in sys.modules

    pre_exec = os.environ.get('GPAW_PREEXEC_SCRIPT')
    if pre_exec is not None:
        import runpy

        runpy.run_path(pre_exec)

    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'

    os.environ['GPAW_MPI_INIT'] = '1'


def main(args=None):
    if args is not None and args.command == 'python':
        # We want the initialization magic to happen first thing,
        # i.e., now, instead of as part of the hook.
        gpaw_python_init_magic()

    from gpaw import __getattr__, all_lazy_imports, broadcast_imports
    with broadcast_imports:
        for attr in all_lazy_imports:
            __getattr__(attr)

        from ase.cli.main import main as ase_main

        from gpaw import __version__

    ase_main('gpaw', 'GPAW command-line tool', __version__,
             commands, hook, args)
