#!/usr/bin/env python3
"""Install gpaw on Niflheim in a virtual environment.

Also installs ase, ase-ext, spglib, sklearn and myqueue.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from sys import version_info

if version_info < (3, 10):
    raise ValueError('Please use Python-3.10 or later')

# Python version in the venv that we are creating
version = '3.13'
fversion = 'cpython-313'

# Niflheim login hosts, with the oldest architecture as the first
nifllogin = [
    'slid2',  # broadwell_el8 (xeon24el8)
    'thul',  # skylake_el8 (xeon40el8, sm3090el8)
    'surt',  # icelake (xeon56, a100, h200)
    'fjorm',  # epyc9004 (epyc96)
    'sara']  # saphirerapids (xeon32)

# Easybuild uses a hierarchy of toolchains for the main foss and intel
# chains.  The order in the tuples before are
#  fullchain: Full chain.
#  mathchain: Chain with math libraries but no MPI
#  compchain: Chain with full compiler suite (but no fancy libs)
#  corechain: Core compiler
# The subchain complementary to 'mathchain', with MPI but no math libs, is
# not used here.

_gcccore = 'GCCcore-14.3.0'
toolchains = {
    'foss': dict(
        fullchain='foss-2025b',
        mathchain='gfbf-2025b',
        compchain='GCC-14.3.0',
        corechain=_gcccore,
    ),
    'intel': dict(
        fullchain='intel-2025b',
        mathchain='iimkl-2025b',
        compchain='intel-compilers-2025.2.0',
        corechain=_gcccore,
    )
}

# These modules are always loaded
module_cmds_all = """\
module purge
unset PYTHONPATH
module load gpaw-data/1.0.1-{corechain}
module load ELPA/2025.06.001-{fullchain}
module load Wannier90/3.1.0-{fullchain}
module load Tkinter/3.13.5-{corechain}
module load libxc/7.0.0-{compchain}
"""

# These modules are not loaded if --piponly is specified
module_cmds_easybuild = """\
module load Python-bundle-PyPI/2025.07-{corechain}
module load matplotlib/3.10.5-{mathchain}
module load scikit-learn/1.7.1-{mathchain}
module load spglib-python/2.6.0-{mathchain}
"""

# These modules are loaded depending on the toolchain
module_cmds_tc = {
    'foss': """\
module load libvdwxc/0.5.0-{fullchain}
""",
    'intel': ""
}

# Arch dependend modules for GPU stuff - not loaded with --piponly
module_cmds_gpu = """\
if [ "$CPU_ARCH" == "icelake" ] && [ {fullchain} == "foss-2025b" ];\
then module load CuPy/13.6.0-{fullchain}-CUDA-12.9.1;fi
if [ "$CPU_ARCH" == "sapphirerapids" ] && [ {fullchain} == "foss-2025b" ];\
then module load CuPy/13.6.0-{fullchain}-CUDA-12.9.1;fi
if [ "$CPU_ARCH" == "skylake_el8" ] && [ {fullchain} == "foss-2025b" ];\
then module load CuPy/13.6.0-{fullchain}-CUDA-12.9.1;fi
if [ "$SLURM_JOB_PARTITION" == "a100" ] \
 || [ "$SLURM_JOB_PARTITION" == "a100_week" ] \
 || [ "$SLURM_JOB_PARTITION" == "sm3090el8" ] \
 || [ "$SLURM_JOB_PARTITION" == "sm3090el8_768" ] \
 || [ "$SLURM_JOB_PARTITION" == "sm3090_devel" ] \
 || [ "$SLURM_JOB_PARTITION" == "h200" ];\
then export GPAW_USE_GPUS=1;export GPAW_NEW=1;fi
"""


activate_extra = """
export GPAW_SETUP_PATH=$GPAW_SETUP_PATH:{venv}/gpaw-basis-pvalence-0.9.20000

# Set matplotlib backend:
if [[ $SLURM_SUBMIT_DIR ]]; then
    export MPLBACKEND=Agg
    export PYTHONWARNINGS="ignore:Matplotlib is currently using agg"
else
    export MPLBACKEND=TkAgg
fi
"""

dftd3 = """\
mkdir {venv}/DFTD3
cd {venv}/DFTD3
URL=https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3
wget $URL/dftd3.tgz
tar -xf dftd3.tgz
ssh {nifllogin[0]} ". {venv}/bin/activate && cd {venv}/DFTD3 && make >& d3.log"
ln -s {venv}/DFTD3/dftd3 {venv}/bin
"""


def run(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    print(cmd)
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def compile_gpaw_c_code(gpaw: Path, activate: Path, intel_only: bool) -> None:
    """Compile for all architectures: xeon24, xeon40, ..."""
    # Remove targets:
    for path in gpaw.glob('build/lib.linux-x86_64-*/_gpaw.*.so'):
        print('Removing', path)
        path.unlink()
    for path in gpaw.glob('niflheim_build/*/_gpaw*.so'):
        print('Removing', path)
        path.unlink()

    # Compile:
    for host in nifllogin:
        if host == 'fjorm' and intel_only:
            continue
        run(f'ssh {host} ". {activate} && pip install -q --no-build-isolation -e {gpaw}"')
        # Save compiled file
        remote_arch = run(f"ssh {host} 'echo $CPU_ARCH'", capture_output=True).stdout.decode().strip()  # Single quote needed in command
        paths = list(gpaw.glob('_gpaw.*.so'))
        assert len(paths) == 1, f'Expected one shared library, found {str(paths)}'
        path = paths[0]
        targetpath = gpaw / 'niflheim_build' / remote_arch
        print(f'Moving {path} to {targetpath}')
        targetpath.mkdir(parents=True, exist_ok=True)
        path.rename(targetpath / path.name)

    # Clean up:
    for path in gpaw.glob('_gpaw.*.so'):
        raise RuntimeError(f'Found unexpected {path}')
    for path in gpaw.glob('build/temp.linux-x86_64-*'):
        shutil.rmtree(path)


def fix_installed_scripts(venvdir: Path,
                          rootdir: str,
                          pythonroot: str) -> None:
    """Fix command line tools so they work in the virtual environment.

    Command line tools (pytest, sphinx-build etc) fail in virtual
    enviroments created with --system-site-packages, as the scripts
    are not copied into the virtual environment.  The scripts have
    the original Python interpreter hardcoded in the hash-bang line.

    This function copies all scripts into the virtual environment,
    and changes the hash-bang so it works.  Starting with the 2023a
    toolchains, the scripts are distributed over more than one
    EasyBuild module.

    Arguments:
    venvdir: Path to the virtual environment
    rootdir: string holding folder of the EasyBuild package being processed
    pythondir: string holding folder of the Python package.
    """

    assert rootdir is not None
    assert pythonroot is not None
    bindir = rootdir / Path('bin')
    print(f'Patching executable scripts from {bindir} to {venvdir}/bin')
    assert '+' not in str(pythonroot) and '+' not in str(venvdir), (
        'Script will fail with "+" in folder names!')
    sedscript = f's+{pythonroot}+{venvdir}+g'

    # Loop over potential executables
    for exe in bindir.iterdir():
        target = venvdir / 'bin' / exe.name
        # Skip files that already exist, are part of Python itself,
        # or are not a regular file or symlink to a file.
        if (not target.exists()
                and not exe.name.lower().startswith('python')
                and exe.is_file()):
            # Check if it is a script file referring the original
            # Python executable in the hash-bang
            with open(exe) as f:
                firstline = f.readline()
            if pythonroot in firstline:
                shutil.copy2(exe, target, follow_symlinks=False)
                # Now patch the file (if not a symlink)
                if not exe.is_symlink():
                    assert not target.is_symlink()
                    subprocess.run(
                        f"sed -e '{sedscript}' --in-place '{target}'",
                        shell=True,
                        check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('venv', help='Name of venv.')
    parser.add_argument('--toolchain', default='foss',
                        choices=['foss', 'intel'],
                        help='Default is foss.')
    parser.add_argument('--dftd3', action='store_true',
                        help='Also build DFTD3.')
    parser.add_argument('--gpaw-branch', default='master',
                        help='Check out a particular gpaw branch')
    parser.add_argument('--run-benchmarks', action='store_true',
                        help='Also submit the GPAW benchmark suite'
                             ' with the checked out branch.')
    parser.add_argument('--recompile', action='store_true',
                        help='Recompile the GPAW C-extensions in an '
                        'exising venv.')
    parser.add_argument('--piponly', action='store_true',
                        help='Do not use EasyBuild python modules, '
                        'install from pip (may affect performance).')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Do not build with GPU support.')
    args = parser.parse_args()

    # if args.toolchain == 'intel':
    #     raise ValueError('See: https://gitlab.com/gpaw/gpaw/-/issues/241')

    venv = Path(args.venv).absolute()
    activate = venv / 'bin/activate'
    gpaw = venv / 'gpaw'

    intel_only = args.toolchain == 'intel'

    def run_benchmarks():
        benchmarkworkflow = gpaw / 'gpaw/benchmark/niflheim-myqueue.py'
        run(f'. {activate} && '
            f'mkdir benchmarks-{args.gpaw_branch} && '
            f'cd benchmarks-{args.gpaw_branch} && '
            f'mq workflow {benchmarkworkflow}')

    if args.recompile:
        compile_gpaw_c_code(gpaw, activate, intel_only)
        if args.run_benchmarks:
            run_benchmarks()

        return 0

    # Sanity checks
    if args.toolchain not in ('foss', 'intel'):
        raise ValueError(f'Unsupported toolchain "{args.toolchain}"')

    module_cmds = module_cmds_all.format(**toolchains[args.toolchain])
    if not args.piponly:
        module_cmds += module_cmds_easybuild.format(
            **toolchains[args.toolchain])
    module_cmds += module_cmds_tc[args.toolchain].format(
        **toolchains[args.toolchain])
    if not args.piponly and not args.no_gpu:
        module_cmds += module_cmds_gpu.format(
            **toolchains[args.toolchain])
    cmds = (' && '.join(module_cmds.splitlines()) +
            f' && python3 -m venv --system-site-packages {args.venv}')
    run(cmds)

    os.chdir(venv)

    activate.write_text(module_cmds +
                        activate.read_text())

    run(f'. {activate} && pip install --upgrade pip -q')

    # Fix venv so pytest etc work
    pythonroot = None
    if args.piponly:
        ebrootvars = ('EBROOTPYTHON',)
    else:
        ebrootvars = ('EBROOTPYTHON', 'EBROOTPYTHONMINBUNDLEMINPYPI')
    for ebrootvar in ebrootvars:
        # Note that we need the environment variable from the newly
        # created venv, NOT from this process!
        comm = run(f'. {activate} && echo ${ebrootvar}',
                   capture_output=True, text=True)
        ebrootdir = comm.stdout.strip()
        if pythonroot is None:
            # The first module is the actual Python module.
            pythonroot = ebrootdir
        assert ebrootdir, f'Env variable {ebrootvar} appears to be unset.'
        fix_installed_scripts(venvdir=venv,
                              rootdir=ebrootdir,
                              pythonroot=pythonroot)

    packages = ['myqueue',
                'graphviz',
                'sphinx_rtd_theme',
                'sphinxcontrib-jquery']
    if args.piponly:
        packages += ['matplotlib',
                     'scipy',
                     'pandas',
                     'pytest',
                     'pytest-xdist',
                     'pytest-mock',
                     'scikit-learn']
    run(f'. {activate} && pip install -q -U ' + ' '.join(packages))

    run('git clone -q https://gitlab.com/ase/ase.git')
    branch = '' if args.gpaw_branch == 'master' else f'-b {args.gpaw_branch} '
    run(f'git clone -q {branch}https://gitlab.com/gpaw/gpaw.git')

    run(f'. {activate} && pip install -q -e ase/')

    if args.dftd3:
        run(' && '.join(dftd3.format(venv=venv,
                                     nifllogin=nifllogin).splitlines()))

    # Compile ase-ext C-extension on old thul so that it works on
    # newer architectures
    run(f'ssh {nifllogin[0]} ". {activate} && pip install -q ase-ext"')

    if args.piponly:
        run('git clone -q https://github.com/spglib/spglib.git')
        run(f'ssh {nifllogin[0]} ". {activate} && pip install {venv}/spglib"')

    # Install GPAW:
    siteconfig = Path(
        f'gpaw/doc/platforms/Linux/Niflheim/siteconfig-{args.toolchain}.py')
    Path('gpaw/siteconfig.py').write_text(siteconfig.read_text())

    compile_gpaw_c_code(gpaw, activate, intel_only)

    for fro, to in [('nahelem', 'icelake'),
                    ('sapphirelake', 'icelake')]:
        f = gpaw / f'niflheim_build/{fro}'
        t = gpaw / f'niflheim_build/{to}'
        f.symlink_to(t)

    # Create .pth file to load correct .so file:
    pth = (
        'import sys, os; '
        'arch = os.environ["CPU_ARCH"]; '
        f"path = f'{venv}/gpaw/niflheim_build/{{arch}}'; "
        'sys.path.append(path)\n')
    Path(f'lib/python{version}/site-packages/niflheim.pth').write_text(pth)

    # Install extra basis-functions:
    run(f'. {activate} && gpaw install-data --basis --version=0.9.20000 '
        f'{venv} --no-register')

    extra = activate_extra.format(venv=venv)

    # Tab completion:
    for cmd in ['ase', 'gpaw', 'mq', 'pip']:
        txt = run(f'. {activate} && {cmd} completion' +
                  (' --bash' if cmd == 'pip' else ''),
                  capture_output=True).stdout.decode()
        extra += txt

    activate.write_text(activate.read_text() + extra)

    # Run tests:
    run(f'. {activate} && ase info && gpaw test')

    if args.run_benchmarks:
        run_benchmarks()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
