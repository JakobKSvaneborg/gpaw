.. _lumi:

=================================
The ``lumi.csc.fi`` supercomputer
=================================

.. note::
   These instructions are up-to-date as of August 2025.

It is recommended to perform the installations under
the ``/projappl/project_...`` directory (see `LUMI storage documentation`_).
A separate installation is needed for LUMI-C and LUMI-G.


Stable GPAW releases
====================

`LUMI software library`_ has EasyBuild recipes for stable GPAW releases.
See `LUMI EasyBuild documentation`_ for detailed description;
steps are only in short below.


Installation on LUMI-G
----------------------

Do the following in a clean terminal session and exit afterwards!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/24.03
  module load partition/G
  module load EasyBuild-user

  # Install GPAW
  eb GPAW-24.6.0-cpeGNU-24.03-rocm.eb -r


Usage on LUMI-G
---------------

Do the following in a clean terminal session (not in the one used for easybuild installations)!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/24.03
  module load partition/G
  module load GPAW/24.6.0-cpeGNU-24.03-rocm
  export MPICH_GPU_SUPPORT_ENABLED=1

  gpaw info


Installation on LUMI-C
----------------------

Do the following in a clean terminal session and exit afterwards!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/24.03
  module load partition/C
  module load EasyBuild-user

  # Install GPAW
  eb GPAW-24.6.0-cpeGNU-24.03.eb -r


Usage on LUMI-C
---------------

Do the following in a clean terminal session (not in the one used for easybuild installations)!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/24.03
  module load partition/C
  module load GPAW/24.6.0-cpeGNU-24.03

  gpaw info


Developer installation
======================

Developer installation on LUMI-G
--------------------------------

For ROCm, it is **strongly** recommended to use the newer ``rocm/6.2.2`` module over the default ``rocm/6.0.3``.
The 6.0.3 module is known to be buggy and cause failures in eg. certain FFT routines. The instructions here are written
for ``rocm/6.2.2`` which generally works better with GPAW.

First, install required libraries as EasyBuild modules
(see `LUMI EasyBuild documentation`_ for detailed description).

Do the following in a clean terminal session and exit afterwards!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/24.03
  module load partition/G
  module load EasyBuild-user

  # Install
  eb CuPy-13.4.1-cpeGNU-24.03-rocm-6.2.2.eb -r
  eb magma-2.8.0-cpeGNU-24.03-rocm6.2.2.eb -r
  eb libxc-7.0.0-cpeGNU-24.03.eb -r

If you need ELPA, an experimental EasyBuild recipe for it that uses ``rocm/6.2.2`` can be found attached in
`this merge request <https://gitlab.com/gpaw/gpaw/-/merge_requests/2724>`_.

Exit the terminal now and open a clean terminal.
The above EasyBuild setup is needed only once.

Then, the following steps build GPAW in a Python virtual environment:

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  cd /projappl/project_.../$USER

  # Create virtual environment
  module load cray-python/3.11.7
  python3 -m venv --system-site-packages venv-gpaw-gpu

  # The following will insert environment setup to the beginning of venv/bin/activate
  cp venv-gpaw-gpu/bin/activate venv-gpaw-gpu/bin/activate.old
  cat << EOF > venv-gpaw-gpu/bin/activate
  export EBU_USER_PREFIX=$EBU_USER_PREFIX
  module load LUMI/24.03
  module load partition/G
  module load cpeGNU/24.03
  module load SuiteSparse/5.13.0-cpeGNU-24.03-OpenMP    # Dependency of hipSolver for ROCm 6.2
  module load rocm/6.2.2
  module load cray-fftw/3.3.10.7
  module load buildtools-python/24.03-cray-python3.11
  module load CuPy/13.4.1-cpeGNU-24.03-rocm-6.2.2       # from EBU_USER_PREFIX
  module load magma/2.8.0-cpeGNU-24.03-rocm6.2.2        # from EBU_USER_PREFIX
  module load libxc/7.0.0-cpeGNU-24.03                  # from EBU_USER_PREFIX
  export MPICH_GPU_SUPPORT_ENABLED=1
  EOF
  cat venv-gpaw-gpu/bin/activate.old >> venv-gpaw-gpu/bin/activate

  # Activate venv
  source venv-gpaw-gpu/bin/activate

  # Freeze the system-provided packages
  pip freeze | tee $(dirname $(which pip))/../constraints.txt

  # Install GPAW development version
  git clone git@gitlab.com:gpaw/gpaw.git
  cd gpaw
  export GPAW_CONFIG=$(readlink -f doc/platforms/Cray/siteconfig-lumi-gpu.py)
  # or:
  # export GPAW_CONFIG=$(readlink -f doc/platforms/Cray/siteconfig-lumi-gpu-elpa.py)
  rm -rf build _gpaw.*.so gpaw.egg-info
  pip install --no-build-isolation --constraint $(dirname $(which pip))/../constraints.txt -v --log build-gpu.log .
  cd ..

Note that above the siteconfig file is taken from the git clone.
Alternatively, download the siteconfig files from here:
:download:`siteconfig-lumi-gpu.py`,
:download:`siteconfig-lumi-gpu-elpa.py`.

For ELPA, remember to also add ``module load ELPA/2024.05.001-cpeGNU-24.03-rocm6.2.2`` in your
``venv-gpaw-gpu/bin/activate`` file.


Usage on LUMI-G
---------------

.. code-block:: bash

  source venv-gpaw-gpu/bin/activate
  gpaw info

Interactive jobs can be run like this::

  srun -p small-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -t 0:30:00 --pty bash

One-liners to run GPU tests::

  n=1; srun   -p small-g --nodes=1 --ntasks-per-node=$n --gpus-per-node=$n -t 00:10:00 gpaw python -m pytest venv-gpaw-gpu/lib/python3.11/site-packages/gpaw/test/ -v -m gpu --basetemp=$PWD/tmp-pytest-gpu-$n --disable-pytest-warnings
  # or:
  n=1; sbatch -p small-g --nodes=1 --ntasks-per-node=$n --gpus-per-node=$n -t 00:10:00 -J pytest-gpu-$n -o %x.out --wrap="srun gpaw python -m pytest venv-gpaw-gpu/lib/python3.11/site-packages/gpaw/test/ -v -m gpu --basetemp=$PWD/tmp-pytest-gpu-$n --disable-pytest-warnings"


Omnitrace
---------

To install `Omnitrace <https://github.com/AMDResearch/omnitrace>`_
(if using custon ROCm, use the correct ROCm version of the installer)::

  cd /projappl/project_...
  wget https://github.com/ROCm/omnitrace/releases/download/rocm-6.2.2/omnitrace-1.11.2-opensuse-15.5-ROCm-60000-PAPI-OMPT-Python3.sh
  bash omnitrace-1.11.2-opensuse-15.5-ROCm-60000-PAPI-OMPT-Python3.sh

To activate Omnitrace, source the env file (after activating GPAW venv)::

  source /projappl/project_.../omnitrace-1.11.2-opensuse-15.5-ROCm-60000-PAPI-OMPT-Python3/share/omnitrace/setup-env.sh


Developer installation on LUMI-C
--------------------------------

First, install required libraries as EasyBuild modules
(see `LUMI EasyBuild documentation`_ for detailed description).

Do the following in a clean terminal session and exit afterwards!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/24.03
  module load partition/C
  module load EasyBuild-user

  # Install
  eb libxc-7.0.0-cpeGNU-24.03.eb -r

Exit the terminal now and open a clean terminal.
The above EasyBuild setup is needed only once.

Then, the following steps build GPAW in a Python virtual environment:

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  cd /projappl/project_.../$USER

  # Create virtual environment
  module load cray-python/3.11.7
  python3 -m venv --system-site-packages venv-gpaw-cpu

  # The following will insert environment setup to the beginning of venv/bin/activate
  cp venv-gpaw-cpu/bin/activate venv-gpaw-cpu/bin/activate.old
  cat << EOF > venv-gpaw-cpu/bin/activate
  export EBU_USER_PREFIX=$EBU_USER_PREFIX
  module load LUMI/24.03
  module load partition/C
  module load cpeGNU/24.03
  module load cray-fftw/3.3.10.7
  module load buildtools-python/24.03-cray-python3.11
  module load libxc/7.0.0-cpeGNU-24.03                  # from EBU_USER_PREFIX
  EOF
  cat venv-gpaw-cpu/bin/activate.old >> venv-gpaw-cpu/bin/activate

  # Activate venv
  source venv-gpaw-cpu/bin/activate

  # Freeze the system-provided packages
  pip freeze | tee $(dirname $(which pip))/../constraints.txt

  # Install GPAW development version
  git clone git@gitlab.com:gpaw/gpaw.git
  cd gpaw
  export GPAW_CONFIG=$(readlink -f doc/platforms/Cray/siteconfig-lumi-cpu.py)
  rm -rf build _gpaw.*.so gpaw.egg-info
  pip install --no-build-isolation --constraint $(dirname $(which pip))/../constraints.txt -v --log build-cpu.log .
  cd ..

Note that above the siteconfig file is taken from the git clone.
Alternatively, download the siteconfig file from here:
:download:`siteconfig-lumi-cpu.py`.


Usage on LUMI-C
---------------

.. code-block:: bash

  source venv-gpaw-cpu/bin/activate
  gpaw info

Interactive jobs can be run like this::

  srun -p small --nodes=1 --ntasks-per-node=2 -t 0:30:00 --pty bash

Two-liner to run tests::

  # Generate gpw files to cache
  srun -p small --nodes=1 --ntasks-per-node=1 --mem-per-cpu=4G -t 01:00:00 gpaw python -m pytest venv-gpaw-cpu/lib/python3.11/site-packages/gpaw/test/test_generate_gpwfiles.py -v -o cache_dir=$PWD/pytest_cache --disable-pytest-warnings
  # Wait and then submit tests
  for n in 1 2 4 8; do sbatch -p small --nodes=1 --ntasks-per-node=$n --mem-per-cpu=4G -t 04:00:00 -J pytest-cpu-$n -o %x.out --wrap="srun gpaw python -m pytest venv-gpaw-cpu/lib/python3.11/site-packages/gpaw/test/ -v -o cache_dir=$PWD/pytest_cache --basetemp=$PWD/tmp-pytest-cpu-$n --disable-pytest-warnings"; done


Configuring MyQueue
===================

Use the following MyQueue_ :file:`config.py` file:

.. literalinclude:: config.py

and submit jobs like this::

  mq submit job.py -R 128:standard:2h

.. _MyQueue: https://myqueue.readthedocs.io/
.. _LUMI storage documentation: https://docs.lumi-supercomputer.eu/storage/
.. _LUMI EasyBuild documentation: https://docs.lumi-supercomputer.eu/software/installing/easybuild/
.. _LUMI software library: https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/
