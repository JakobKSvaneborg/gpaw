.. _Mahti:

====================
Mahti (mahti.csc.fi)
====================

.. note::
   These instructions are up-to-date as of July 2025.

Stable GPAW releases
====================

Mahti has several versions of GPAW available as modules. You can load the
most recent module with ``module load gpaw``, and browse other available versions
with ``module spider gpaw``.

Currently the pre-installed modules have **CPU support only**.
A manual (developer) installation is needed for GPU support.

Developer installation on Mahti
===============================

GPAW for CPUs
-------------

Do the following in a new terminal session.

.. code-block:: bash

    # Move to installation directory of your choice
    cd /projappl/project_.../$USER

    # Use a specific Python installation
    export python=/appl/spack/v017/views/gpaw-python3.9/bin/python3.9

    # Create virtual environment
    $python -m venv --system-site-packages venv-gpaw-cpu

    # The following will insert environment setup to the beginning of venv/bin/activate
    cp venv-gpaw-cpu/bin/activate venv-gpaw-cpu/bin/activate.old
    cat << EOF > venv-gpaw-cpu/bin/activate
    module load gcc/11.2.0
    module load openmpi/4.1.2
    module load openblas/0.3.18-omp
    module load fftw/3.3.10-mpi
    module load netlib-scalapack/2.1.0
    EOF
    cat venv-gpaw-cpu/bin/activate.old >> venv-gpaw-cpu/bin/activate

    # Activate venv
    source venv-gpaw-cpu/bin/activate

    # Clone GPAW development repository
    git clone https://gitlab.com/gpaw/gpaw.git
    cd gpaw
    rm -rf build _gpaw.*.so gpaw.egg-info

    export GPAW_CONFIG=$(readlink -f doc/platforms/Linux/Mahti/siteconfig-mahti-cpu.py)

    # Install GPAW, with a constraint to ensure we use system-provided packages.
    # Leave the '-e' out if you don't want an editable install
    pip install --no-build-isolation --constraint $(dirname $(which pip))/../constraints.txt -v --log build-cpu.log -e .

The above gets ``siteconfig.py`` from the cloned Git repository.
Alternatively, you can download it from here:
:download:`siteconfig-mahti-cpu.py`,
