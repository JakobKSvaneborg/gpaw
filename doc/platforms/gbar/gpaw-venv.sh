#!/usr/bin/bash
# Install gpaw, ase, ase-ext, spglib, scikit-learn and myqueue in a venv

set -e  # stop if there are errors

NAME=$1
FOLDER=$PWD

echo "
# module purge does not work as dcc-setup/2023-aug is sticky
module unload `module -t list | grep -v 'Currently Loaded' | grep -v dcc-setup`
source /dtu/sw/dcc/dcc-sw.bash
module load python/3.11.7
module load fftw/3.3.10 libxc/6.1.0
module load scalapack/2.2.0
module load openblas/0.3.28
module load dftd3/3.2.0
unset CC
" > modules.sh

. modules.sh

# Create venv:
echo "Creating virtual environment $NAME"
python3 -m venv --system-site-packages $NAME
cd $NAME
VENV=$PWD
. bin/activate
PIP="python3 -m pip"
$PIP install --upgrade pip -qq

# Load modules in activate script:
mv bin/activate old
mv ../modules.sh bin/activate
cat old >> bin/activate
rm old

# Fix missing dependency in preinstalled packages
$PIP install pandas

# Install ASE from git:
git clone https://gitlab.com/ase/ase.git
$PIP install -e ase/

$PIP install myqueue graphviz ase-ext spglib scikit-learn pytest-xdist

# Install GPAW:
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
echo "
from pathlib import Path
from os import environ as env
scalapack = True
fftw = True
libraries = ['xc', 'openblas', 'fftw3', 'scalapack']
base = Path(env['DCC_SW_PATH']) / env['DCC_SW_CPUTYPE'] / env['DCC_SW_COMPILER']

for p in ['fftw/3.3.10', 'libxc/6.1.0', 'scalapack/2.2.0', 'openblas/0.3.28']:
    lib = base / p / 'lib'
    library_dirs.append(lib)
    extra_link_args.append(f'-Wl,-rpath={lib}')
    include_dirs.append(f'{base / p}/include')
" > siteconfig.py
pip install -e . -v > gpaw.out 2>&1

# Install extra basis-functions:
cd $VENV
export GPAW_SETUP_PATH=~jasc/PAW/gpaw-setups-24.11.0/
echo "export GPAW_SETUP_PATH=$GPAW_SETUP_PATH" >> bin/activate

# Tab completion:
ase completion >> bin/activate
gpaw completion >> bin/activate
mq completion >> bin/activate

# Run tests:
mq --version
ase info
gpaw test
