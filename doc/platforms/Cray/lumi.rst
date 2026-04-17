.. _lumi:

=================================
The ``lumi.csc.fi`` supercomputer
=================================

.. note::
   These instructions are up-to-date as of April 2026.

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
  module load LUMI/25.03
  module load partition/G
  module load EasyBuild-user

  # Install GPAW
  eb GPAW-25.7.0-cpeGNU-25.03-rocm.eb -r

  # Exit the terminal after easybuild installations!
  exit


Usage on LUMI-G
---------------

Do the following in a clean terminal session (not in the one used for easybuild installations)!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/25.03
  module load partition/G
  module load GPAW/25.7.0-cpeGNU-25.03-rocm
  export MPICH_GPU_SUPPORT_ENABLED=1

  gpaw info


Installation on LUMI-C
----------------------

Do the following in a clean terminal session and exit afterwards!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/25.03
  module load partition/C
  module load EasyBuild-user

  # Install GPAW
  eb GPAW-25.7.0-cpeGNU-25.03.eb -r

  # Exit the terminal after easybuild installations!
  exit


Usage on LUMI-C
---------------

Do the following in a clean terminal session (not in the one used for easybuild installations)!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/25.03
  module load partition/C
  module load GPAW/25.7.0-cpeGNU-25.03

  gpaw info


Developer installation
======================

Developer installation on LUMI-G
--------------------------------

First, install required libraries as EasyBuild modules
(see `LUMI EasyBuild documentation`_ for detailed description).

Do the following in a clean terminal session and exit afterwards!

.. code-block:: bash

  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/25.03
  module load partition/G
  module load EasyBuild-user

  # Install
  eb CuPy-13.5.1-cpeGNU-25.03-rocm.eb -r
  eb magma-2.9.0-cpeAMD-25.03-rocm.eb -r
  eb libxc-7.0.0-cpeGNU-25.03-FHC.eb -r

  # Exit the terminal after easybuild installations!
  exit


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
  module load LUMI/25.03
  module load partition/G
  module load cpeGNU/25.03
  module load rocm/6.3.4
  module load cray-fftw/3.3.10.10
  module load buildtools-python/25.03-cray-python3.11
  module load CuPy/13.5.1-cpeGNU-25.03-rocm             # from EBU_USER_PREFIX
  module load magma/2.9.0-cpeGNU-25.03-rocm             # from EBU_USER_PREFIX
  module load libxc/7.0.0-cpeGNU-25.03-FHC              # from EBU_USER_PREFIX
  export MPICH_GPU_SUPPORT_ENABLED=1
  export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
  export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)
  EOF
  cat venv-gpaw-gpu/bin/activate.old >> venv-gpaw-gpu/bin/activate

  # Activate venv
  source venv-gpaw-gpu/bin/activate

  # Update build tools
  pip install --upgrade pip setuptools packaging pybind11

  # Freeze the system-provided packages
  pip list --format=freeze | tee $(dirname $(which pip))/../constraints.txt

  # Clone GPAW development repository
  git clone https://gitlab.com/gpaw/gpaw.git
  cd gpaw

  export GPAW_CONFIG=$(readlink -f doc/platforms/Cray/siteconfig-lumi-gpu.py)
  # or:
  # export GPAW_CONFIG=$(readlink -f doc/platforms/Cray/siteconfig-lumi-gpu-elpa.py)

  # Install GPAW, with a constraint to ensure we use system-provided packages.
  # Leave the '-e' out if you don't want an editable install
  rm -rf _build build _gpaw.*.so gpaw.egg-info
  GPAW_BUILD_JOBS=16 pip install --no-build-isolation --constraint $(dirname $(which pip))/../constraints.txt -v --log build-gpu.log -e .
  cd ..

Note that above the siteconfig file is taken from the git clone.
Alternatively, download the siteconfig files from here:
:download:`siteconfig-lumi-gpu.py`,
:download:`siteconfig-lumi-gpu-elpa.py`.


Usage on LUMI-G
---------------

.. code-block:: bash

  source venv-gpaw-gpu/bin/activate
  gpaw info

Interactive jobs can be run like this::

  srun -p small-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -t 0:30:00 --pty bash

To run GPU tests::

  # Run in an empty directory
  mkdir run
  cd run

  # Find GPAW python files
  GPAW_HOME=$(dirname $(python -c 'import gpaw; print(gpaw.__file__)' | head -n 1))

  n=1; srun -p small-g --nodes=1 --ntasks-per-node=$n --gpus-per-node=$n -t 00:10:00 python -m pytest $GPAW_HOME/test/ -v -m gpu --basetemp=$PWD/tmp-pytest-gpu-$n --disable-pytest-warnings
  # or:
  n=1; sbatch -p small-g --nodes=1 --ntasks-per-node=$n --gpus-per-node=$n -t 00:10:00 -J pytest-gpu-$n -o %x.out --wrap="srun python -m pytest $GPAW_HOME/test/ -v -m gpu --basetemp=$PWD/tmp-pytest-gpu-$n --disable-pytest-warnings"


Omnitrace
---------

To install `Omnitrace <https://github.com/AMDResearch/omnitrace>`_
(if using custom ROCm, use the correct ROCm version of the installer)::

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
  module load LUMI/25.03
  module load partition/C
  module load EasyBuild-user

  # Install
  eb libxc-7.0.0-cpeGNU-25.03-FHC.eb -r

  # Exit the terminal after easybuild installations!
  exit


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
  module load LUMI/25.03
  module load partition/C
  module load cpeGNU/25.03
  module load cray-fftw/3.3.10.10
  module load buildtools-python/25.03-cray-python3.11
  module load libxc/7.0.0-cpeGNU-25.03-FHC              # from EBU_USER_PREFIX
  EOF
  cat venv-gpaw-cpu/bin/activate.old >> venv-gpaw-cpu/bin/activate

  # Activate venv
  source venv-gpaw-cpu/bin/activate

  # Update build tools
  pip install --upgrade pip setuptools packaging pybind11

  # Freeze the system-provided packages
  pip list --format=freeze | tee $(dirname $(which pip))/../constraints.txt

  # Clone GPAW development repository
  git clone https://gitlab.com/gpaw/gpaw.git
  cd gpaw

  export GPAW_CONFIG=$(readlink -f doc/platforms/Cray/siteconfig-lumi-cpu.py)

  # Install GPAW, with a constraint to ensure we use system-provided packages.
  # Leave the '-e' out if you don't want an editable install
  rm -rf _build build _gpaw.*.so gpaw.egg-info
  GPAW_BUILD_JOBS=16 pip install --no-build-isolation --constraint $(dirname $(which pip))/../constraints.txt -v --log build-cpu.log -e .
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

To run tests::

  # Run in an empty directory
  mkdir run
  cd run

  # Find GPAW python files
  GPAW_HOME=$(dirname $(python -c 'import gpaw; print(gpaw.__file__)' | head -n 1))

  # Generate gpw files to cache
  srun -p small --nodes=1 --ntasks-per-node=1 --mem-per-cpu=4G -t 01:00:00 python -m pytest $GPAW_HOME/test/test_generate_gpwfiles.py -v -o cache_dir=$PWD/pytest_cache --disable-pytest-warnings

  # Wait and then submit tests
  for n in 1 2 4 8; do sbatch -p small --nodes=1 --ntasks-per-node=$n --mem-per-cpu=4G -t 04:00:00 -J pytest-cpu-$n -o %x.out --wrap="srun python -m pytest $GPAW_HOME/test/ -v -o cache_dir=$PWD/pytest_cache --basetemp=$PWD/tmp-pytest-cpu-$n --disable-pytest-warnings"; done


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
