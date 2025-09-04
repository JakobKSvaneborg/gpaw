.. _Puhti:

====================
Puhti (puhti.csc.fi)
====================

.. note::
   These instructions are up-to-date as of August 2025.

Stable GPAW releases
====================

Puhti has several versions of GPAW available as modules. You can load the
most recent module with ``module load gpaw``, and browse other available versions
with ``module spider gpaw``.

Currently the most up-to-date module is for GPAW version 24.6.0 with **CPU support only**.
A manual installation is needed for GPU support.

Developer installation on Puhti
============================

GPAW for CPUs
-------------

Do the following in a new terminal session.

.. code-block:: bash

    # Move to installation directory of your choice
    cd /projappl/project_.../$USER

    # Use a specific Python installation
    export python=/appl/spack/v018/views/gpaw-python-env/bin/python3.9

    # Create virtual environment
    $python -m venv --system-site-packages venv-gpaw-cpu

    # The following will insert environment setup to the beginning of venv/bin/activate
    cp venv-gpaw-cpu/bin/activate venv-gpaw-cpu/bin/activate.old
    cat << EOF > venv-gpaw-cpu/bin/activate
    module load gcc/11.3.0
    module load openmpi/4.1.4
    module load intel-oneapi-mkl/2022.1.0
    module load fftw/3.3.10-mpi-omp
    EOF
    cat venv-gpaw-cpu/bin/activate.old >> venv-gpaw-cpu/bin/activate

    # Activate venv
    source venv-gpaw-cpu/bin/activate

    # Clone GPAW development repository
    git clone https://gitlab.com/gpaw/gpaw.git
    cd gpaw
    rm -rf build _gpaw.*.so gpaw.egg-info

    export GPAW_CONFIG=$(readlink -f doc/platforms/Linux/Puhti/siteconfig-puhti-cpu.py)

    # Install GPAW. Leave out '-e' if you don't want an editable install
    pip install -v --log build-cpu.log -e .

The above gets ``siteconfig.py`` from the cloned Git repository.
Alternatively, you can download it from here:
:download:`siteconfig-puhti-cpu.py`,
