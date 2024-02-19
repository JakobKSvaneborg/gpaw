.. _lumi:

=================================
The ``lumi.csc.fi`` supercomputer
=================================

.. note::
   These instructions are up-to-date as of November 2023.

It is recommended to perform the installations under
the ``/projappl/project_...`` directory
(see `LUMI user documentation <https://docs.lumi-supercomputer.eu/storage/>`_).
A separate installation is needed for LUMI-C and LUMI-G.


GPAW for LUMI-G
===============

First, install required libraries as EasyBuild modules
(see `LUMI user documentation <https://docs.lumi-supercomputer.eu/software/installing/easybuild/>`_
for detailed description):

.. code-block:: bash

  # Setup environment
  # TODO: use correct project_...
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  module load LUMI/22.12 partition/G
  module load cpeGNU/22.12
  module load rocm/5.2.3
  module load EasyBuild-user

  # Install CuPy
  eb CuPy-12.2.0-cpeGNU-22.12.eb -r

  # Install libxc
  eb libxc-6.2.2-cpeGNU-22.12.eb -r

The above EasyBuild setup is needed only once.

Then, the following steps build GPAW in a Python virtual environment:

.. code-block:: bash

  # Create virtual environment
  module load cray-python/3.9.13.1
  python3 -m venv --system-site-packages venv

  # The following will insert environment setup to the beginning of venv/bin/activate
  # TODO: use correct project_...
  cp venv/bin/activate venv/bin/activate.old
  cat << EOF > venv/bin/activate
  export EBU_USER_PREFIX=/projappl/project_.../EasyBuild
  export GPAW_SETUP_PATH=/projappl/project_.../gpaw-setups-0.9.20000
  module load LUMI/22.12 partition/G
  module load cpeGNU/22.12
  module load rocm/5.2.3
  module load cray-python/3.9.13.1
  module load cray-fftw/3.3.10.1
  module load CuPy/12.2.0-cpeGNU-22.12  # from EBU_USER_PREFIX
  module load libxc/6.2.2-cpeGNU-22.12  # from EBU_USER_PREFIX
  EOF
  cat venv/bin/activate.old >> venv/bin/activate

  # Activate venv
  source venv/bin/activate

  # Install GPAW development version
  git clone git@gitlab.com:gpaw/gpaw.git
  export GPAW_CONFIG=$(readlink -f gpaw/doc/platforms/Cray/siteconfig-lumi-gpu.py)
  cd gpaw
  rm -rf build _gpaw.*.so gpaw.egg-info
  pip install -v --log build.log -e .

  # Install gpaw setups
  # TODO: use correct project_...
  gpaw install-data --no-register /projappl/project_...

Note that above the siteconfig file is taken from the git clone.
If you are not using installation through git, use the siteconfig file from here:
:git:`~doc/platforms/Cray/siteconfig-lumi-gpu.py`.

Interactive jobs can be run like this::

  srun -A project_... -p small-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=1 -t 0:30:00 --pty bash

Omnitrace
---------

To install `Omnitrace <https://github.com/AMDResearch/omnitrace>`_
(if using custon ROCm, use the correct ROCm version of the installer)::

  cd /projappl/project_...
  wget https://github.com/AMDResearch/omnitrace/releases/download/v1.10.4/omnitrace-1.10.4-opensuse-15.4-ROCm-50200-PAPI-OMPT-Python3.sh
  bash omnitrace-1.10.4-opensuse-15.4-ROCm-50200-PAPI-OMPT-Python3.sh

To activate Omnitrace, source the env file (after activating GPAW venv)::

  source /projappl/project_.../omnitrace-1.10.4-opensuse-15.4-ROCm-50200-PAPI-OMPT-Python3/share/omnitrace/setup-env.sh


Configuring MyQueue
===================

Use the following MyQueue_ :file:`config.py` file:

.. literalinclude:: config.py

and submit jobs like this::

  mq submit job.py -R 128:standard:2h

.. _MyQueue: https://myqueue.readthedocs.io/en/latest/
