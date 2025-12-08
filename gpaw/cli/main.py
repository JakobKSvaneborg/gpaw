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

        if args.parallel is None:
            # Let MPI decide nprocs when not given.  We use 0 to distinguish
            # that value.  We cannot use None because that may be used
            # by other commands that can run in parallel.
            args.parallel = 0

    if hasattr(args, 'dry_run'):
        N = int(args.dry_run)
        if N:
            import gpaw
            gpaw.dry_run = N
            import gpaw.mpi as mpi
            mpi.world = mpi.SerialCommunicator()
            mpi.world.size = N

    if args.parallel is not None:
        from gpaw.mpi import compiled_with_mpi, world, have_mpi

        if not compiled_with_mpi:
            raise SystemExit('MPI not available')

        if have_mpi:
            # When the user runs "gpaw -Pn python" then that's in serial
            # but it ends up calling itself recursively (!) after adding
            # some MPI options.  When it runs the second time (actually in
            # parallel) we set the GPAW_MPI_INIT variable.  If that envvar is
            # set, then we are already parallel and this hook can return.
            # Otherwise, we need to start the parallel subprocess.
            return args

        # Start again in parallel:
        pyargs = []
        if sys.version_info >= (3, 11):
            # Don't prepend a potentially unsafe path to sys.path
            pyargs.append('-P')

        mpiargs = os.environ.get('GPAW_MPI_OPTIONS', '').split()
        if args.parallel != 0:
            mpiargs += ['-n', str(args.parallel)]

        arguments = ['mpiexec',
                     *mpiargs,
                     sys.executable,
                     *pyargs,
                     '-m',
                     'gpaw',
                     *sys.argv[1:]]

        # Use a clean set of environment variables without any MPI stuff:
        env = dict(os.environ)
        env['GPAW_MPI_INIT'] = '1'
        if 'OMP_NUM_THREADS' not in env:
            env['OMP_NUM_THREADS'] = '1'

        p = subprocess.run(arguments, check=False, env=env)
        sys.exit(p.returncode)

    return args


def gpaw_python_init_magic():
    # We run this very early so as to set required environment variables
    # before anybody else gets to see them.
    pre_exec = os.environ.get('GPAW_PREEXEC_SCRIPT')
    if pre_exec is not None:
        import runpy

        runpy.run_path(pre_exec)

    from gpaw import __getattr__, all_lazy_imports, broadcast_imports
    with broadcast_imports:
        for attr in all_lazy_imports:
            __getattr__(attr)

        from ase.cli.main import main as ase_main
        from gpaw.mpi import world
        from gpaw import __version__

    from ase.parallel import world as ase_world
    assert ase_world.size == world.size


def main(args=None):
    if os.environ.get('GPAW_MPI_INIT'):
        gpaw_python_init_magic()

    from ase.cli.main import main as ase_main
    from gpaw import __version__
    ase_main('gpaw', 'GPAW command-line tool', __version__,
             commands, hook, args)
