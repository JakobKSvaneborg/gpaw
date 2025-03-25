#!/usr/bin/bash
set -e  # stop if there are errors
NAME=$1
FOLDER=$PWD

echo '
export EASYBUILD_PREFIX=/groups/physics/modules
module use $EASYBUILD_PREFIX/modules/all
module purge
unset PYTHONPATH
module load GPAW/21.6.0-foss-2020b-libxc-5.1.5-ASE-3.22.0
' > modules.sh
. modules.sh

python3 -m venv $NAME
cd $NAME
VENV=$PWD
. bin/activate
PIP="python3 -m pip"
$PIP install --upgrade pip -qq

mv bin/activate old
mv ../modules.sh bin/activate
cat old >> bin/activate
rm old
$PIP install pytz
git clone https://gitlab.com/ase/ase.git
$PIP install -U ase/

git clone https://gitlab.com/asr-dev/asr.git
cd asr
git checkout old-master
#git checkout om_spinspiral
cd ..
$PIP install -e asr

git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
#git checkout old-spinspiral
cd ..
echo "
from os import environ
from pathlib import Path
scalapack = True
fftw = True
libraries = ['openblas', 'fftw3', 'readline', 'gfortran',
             'scalapack', 'xc', 'vdwxc']
libxc = Path(environ['EBROOTLIBXC'])
include_dirs.append(libxc / 'include')
libvdwxc = Path(environ['EBROOTLIBVDWXC'])
include_dirs.append(libvdwxc / 'include')
library_dirs = environ['LD_LIBRARY_PATH'].split(':')
" > gpaw/siteconfig.py
pip install -e gpaw

$PIP install myqueue graphviz qeh ase-ext

gpaw install-data --basis --version=0.9.20000 . --no-register
export GPAW_SETUP_PATH=$GPAW_SETUP_PATH:$VENV/gpaw-basis-pvalence-0.9.20000
echo "export GPAW_SETUP_PATH=$GPAW_SETUP_PATH" >> bin/activate

ase completion >> bin/activate
gpaw completion >> bin/activate
mq completion >> bin/activate
$PIP completion --bash >> bin/activate

echo '
if [[ $SLURM_SUBMIT_DIR ]]; then
    export MPLBACKEND=Agg
else
    export MPLBACKEND=TkAgg
fi' >> bin/activate

pip install flask==2.1.0

# Fix to remove the GPAW and ASE module installed in the "GPAW/21.6.0-foss-2020b" toolchain from path
echo '
export PYTHONPATH=/groups/physics/modules/software/spglib-python/1.16.0-foss-2020b/lib/python3.8/site-packages:/groups/physics/modules/software/matplotlib/3.3.3-foss-2020b/lib/python3.8/site-packages:/g\
roups/physics/modules/software/Pillow/8.0.1-GCCcore-10.2.0/lib/python3.8/site-packages:/groups/physics/modules/software/Tkinter/3.8.6-GCCcore-10.2.0/lib/python3.8:/groups/physics/modules/software/Tkinte\
r/3.8.6-GCCcore-10.2.0/easybuild/python:/groups/physics/modules/software/SciPy-bundle/2020.11-foss-2020b/lib/python3.8/site-packages:/groups/physics/modules/software/pybind11/2.6.0-GCCcore-10.2.0/lib/py\
thon3.8/site-packages:/groups/physics/modules/software/Python/3.8.6-GCCcore-10.2.0/easybuild/python
' >> bin/activate

mq --version
#ase info
#gpaw test
source bin/activate
pip uninstall ase
pip install ase/
ase info
gpaw test
