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

if version_info < (3, 7):
    raise ValueError('Please use Python-3.7 or later')

version = '3.10'  # Python version in the venv that we are creating

module_cmds_all = f"""\
module purge
unset PYTHONPATH
module load matplotlib/3.5.2-{{tchain}}-2022a
module load GPAW-setups/0.9.20000
module load Wannier90/3.1.0-{{tchain}}-2022a
module load ELPA/2022.05.001-{{tchain}}-2022a
module load scikit-learn/1.1.2-{{tchain}}-2022a
"""

module_cmds_tc = {
    'foss': """\
module load libxc/5.2.3-GCC-11.3.0
module load libvdwxc/0.4.0-foss-2022a
""",
    'intel': """\
module load libxc/5.2.3-intel-compilers-2022.1.0
"""}

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
URL=https://www.chemiebn.uni-bonn.de/pctc/mulliken-center/software/dft-d3
wget $URL/dftd3.tgz
tar -xf dftd3.tgz
ssh thul ". {venv}/bin/activate && cd {venv}/DFTD3 && make"
ln -s {venv}/DFTD3/dftd3 {venv}/bin
"""


def run(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    print(cmd)
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def compile_gpaw_c_code(gpaw: Path, activate: Path) -> None:
    """Compile for all architectures: xeon16, xeon24, xeon40, ..."""
    # Remove targets:
    for path in gpaw.glob('build/lib.linux-x86_64-*/_gpaw.*.so'):
        path.unlink()

    # Compile:
    for host in ['thul', 'sylg', 'svol', 'surt']:
        run(f'ssh {host} ". {activate} && pip install -q -e {gpaw}"')

    # Clean up:
    for path in gpaw.glob('_gpaw.*.so'):
        path.unlink()
    for path in gpaw.glob('build/temp.linux-x86_64-*'):
        shutil.rmtree(path)

def fix_installed_scripts(venvdir: Path) -> None:
    """Fix command line tools so they work in the virtual environment.

    Command line tools (pytest, sphinx-build etc) fail in virtual
    enviroments created with --system-site-packages, as the scripts
    are not copied into the virtual environment.  The scripts have
    the original Python interpreter hardcoded in the hash-bang line.

    This function copies all scripts into the virtual environment,
    and changes the hash-bang so it works.
    """

    rootdir = os.getenv('EBROOTPYTHON')
    bindir = rootdir / Path('bin')
    print(f'Patching binaries from {bindir} to {venvdir}/bin')
    sedscript = f's+{rootdir}+{venvdir}+g'
    #print('sed script:', sedscript)

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
            if rootdir in firstline:
                shutil.copy2(exe, target, follow_symlinks=False)
                # Now patch the file (if not a symlink)
                if not exe.is_symlink():
                    assert not target.is_symlink()
                    subprocess.run(
                        f"sed -e '{sedscript}' --in-place '{target}'",
                        shell=True,
                        check=True
                    )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('venv', help='Name of venv.')
    parser.add_argument('--toolchain', default='foss',
                        choices=['foss', 'intel'],
                        help='Default is foss.')
    parser.add_argument('--dftd3', action='store_true',
                        help='Also build DFTD3.')
    parser.add_argument('--recompile', action='store_true',
                        help='Recompile the GPAW C-extensions in an '
                        'exising venv.')
    args = parser.parse_args()

    #if args.toolchain == 'intel':
    #    raise ValueError('See: https://gitlab.com/gpaw/gpaw/-/issues/241')

    venv = Path(args.venv).absolute()
    activate = venv / 'bin/activate'
    gpaw = venv / 'gpaw'

    if args.recompile:
        compile_gpaw_c_code(gpaw, activate)
        return 0

    module_cmds = module_cmds_all.format(tchain=args.toolchain)
    module_cmds += module_cmds_tc[args.toolchain]

    cmds = (' && '.join(module_cmds.splitlines()) +
            f' && python3 -m venv --system-site-packages {args.venv}')
    run(cmds)

    os.chdir(venv)

    activate.write_text(module_cmds +
                        activate.read_text())

    run(f'. {activate} && pip install --upgrade pip -q')

    # Fix venv so pytest etc work
    fix_installed_scripts(venvdir=venv)

    packages = ['myqueue',
                'graphviz',
                'pytest',
                'pytest-xdist',
                'qeh',
                'sphinx_rtd_theme']
    run(f'. {activate} && pip install -q -U ' + ' '.join(packages))

    for name in ['ase', 'gpaw']:
        run(f'git clone -q https://gitlab.com/{name}/{name}.git')

    run(f'. {activate} && pip install -q -e ase/')

    if args.dftd3:
        run(' && '.join(dftd3.format(venv=venv).splitlines()))

    # Compile ase-ext C-extension on old thul so that it works on
    # newer architectures
    run(f'ssh thul ". {activate} && pip install -q ase-ext"')

    run('git clone -q https://github.com/spglib/spglib.git')
    run(f'ssh thul ". {activate} && pip install {venv}/spglib"')

    # Install GPAW:
    siteconfig = Path(
        f'gpaw/doc/platforms/Linux/Niflheim/siteconfig-{args.toolchain}.py')
    Path('gpaw/siteconfig.py').write_text(siteconfig.read_text())

    compile_gpaw_c_code(gpaw, activate)

    for fro, to in [('ivybridge', 'sandybridge'),
                    ('nahelem', 'icelake')]:
        f = gpaw / f'build/lib.linux-x86_64-{fro}-{version}'
        t = gpaw / f'build/lib.linux-x86_64-{to}-{version}'
        f.symlink_to(t)

    # Create .pth file to load correct .so file:
    pth = ('import sys, os; '
           'arch = os.environ["CPU_ARCH"]; '
           f"path = f'{venv}/gpaw/build/lib.linux-x86_64-{{arch}}-{version}'; "
           'sys.path.append(path)\n')
    Path(f'lib/python{version}/site-packages/niflheim.pth').write_text(pth)

    # Install extra basis-functions:
    run(f'. {activate} && gpaw install-data --basis --version=20000 '
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

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
