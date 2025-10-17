.. _Mahti:

====================
Mahti (mahti.csc.fi)
====================

.. note::
   These instructions are up-to-date as of September 2025.

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
    export python=/appl/spack/v023/views/gpaw-python-311/bin/python3.11

    # Create virtual environment
    $python -m venv --system-site-packages venv-gpaw-cpu

    # The following will insert environment setup to the beginning of venv/bin/activate
    cp venv-gpaw-cpu/bin/activate venv-gpaw-cpu/bin/activate.old
    cat << EOF > venv-gpaw-cpu/bin/activate
    module load gcc/14.2.0
    module load openmpi/5.0.6
    module load openblas/0.3.28-omp
    module load fftw/3.3.10-mpi-omp
    module load netlib-scalapack/2.2.0

    # Set library and include paths for non-module dependencies (xc)
    export LIBRARY_PATH=/appl/spack/v023/views/gpaw-python-311/lib64:/appl/spack/v023/views/gpaw-python-311/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/appl/spack/v023/views/gpaw-python-311/lib64:/appl/spack/v023/views/gpaw-python-311/lib:$LD_LIBRARY_PATH
    export CPATH=/appl/spack/v023/views/gpaw-python-311/include:$CPATH

    # Paths for libvdwxc separately. We use the v0.4.0 install from gcc-13.1.0 tree which was built for OpenMPI-4.
    # Not ideal but works (ABIs are compatible)
    export LIBRARY_PATH=/appl/spack/v020/install-tree/gcc-13.1.0/libvdwxc-0.4.0-5vlzlb/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/appl/spack/v020/install-tree/gcc-13.1.0/libvdwxc-0.4.0-5vlzlb/lib:$LD_LIBRARY_PATH
    export CPATH=/appl/spack/v020/install-tree/gcc-13.1.0/libvdwxc-0.4.0-5vlzlb/include:$CPATH
    EOF
    cat venv-gpaw-cpu/bin/activate.old >> venv-gpaw-cpu/bin/activate

    # Activate venv
    source venv-gpaw-cpu/bin/activate

    # Clone GPAW development repository
    git clone https://gitlab.com/gpaw/gpaw.git
    cd gpaw

    export GPAW_CONFIG=$(readlink -f doc/platforms/Linux/Mahti/siteconfig-mahti-cpu.py)

    # Install GPAW. Leave the '-e' out if you don't want an editable install
    rm -rf build _gpaw.*.so gpaw.egg-info
    pip install --no-build-isolation -v --log build-cpu.log -e .

The above gets ``siteconfig.py`` from the cloned Git repository.
Alternatively, you can download it from here:
:download:`siteconfig-mahti-cpu.py`,
