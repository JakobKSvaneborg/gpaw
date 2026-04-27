#!/usr/bin/bash
# Install gpaw, ase, ase-ext on Juwels in a virtual environment

set -e  # stop if there are errors

NAME="$1"
USAGE="Usage: $0 foldername"
FOLDER="$PWD"
ASE_REPO=https://gitlab.com/ase/ase.git
GPAW_REPO=https://gitlab.com/gpaw/gpaw.git

if [[ $# -ne 1 ]]; then
    echo "Wrong number of arguments, expected 1 (install dir), got $#"
    echo $USAGE
    exit 1
fi

echo "
module purge
module  load StdEnv
module load Python
module load SciPy-Stack
module load intel-para/2021
module load FFTW
module load libxc
module load ELPA
" > modules.sh

. modules.sh

# Create venv:
echo "Creating virtual environment \"$NAME\""
python3 -m venv --system-site-packages "$NAME"
cd "$NAME"
VENV="$PWD"
. bin/activate
PIP="python3 -m pip"
$PIP install --upgrade pip -qq

# Load modules in activate script:
mv bin/activate old
mv "$FOLDER/modules.sh" bin/activate
cat old >> bin/activate
rm old

# Install ASE from git:
git clone $ASE_REPO
$PIP install -e ase/

$PIP install myqueue graphviz qeh

CMD="cd \"$VENV\" &&
     . bin/activate &&
     pip install ase-ext"
echo "$CMD"

# Install GPAW:
git clone $GPAW_REPO
cd gpaw
cp ./doc/platforms/Linux/Juwels/siteconfig_juwels.py siteconfig.py
cd "$VENV"
. bin/activate
pip install -e gpaw -v > compilation.out

# Install extra basis-functions:
cd "$VENV"
gpaw install-data --basis --version=0.9.20000 . --no-register
export GPAW_SETUP_PATH="$GPAW_SETUP_PATH:$VENV/gpaw-basis-pvalence-0.9.20000"
echo "export GPAW_SETUP_PATH=$GPAW_SETUP_PATH" >> bin/activate

# Tab completion:
ase completion >> bin/activate
gpaw completion >> bin/activate
mq completion >> bin/activate
$PIP completion --bash >> bin/activate

# Set matplotlib backend:
echo '
if [[ $SLURM_SUBMIT_DIR ]]; then
    export MPLBACKEND=Agg
else
    export MPLBACKEND=TkAgg
fi
' >> bin/activate

# Run tests:
mq --version
ase info
gpaw test
