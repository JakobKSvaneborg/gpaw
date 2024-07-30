.. _nersc_perlmuter:

==============================
perlmutter.nersc.gov (Cray EX)
==============================

.. note::
   These instructions are up-to-date as of Feburary 2024.

GPAW
====


Prerequisite
------------

This version of GPAW install depends on libxc version 6.2.2. First install
libxc and then edit the file ``siteconfig.py`` to point to the location of
the install. An example ``siteconfig.py`` is included at the end of these
directions.

Install
-------

Installing GPAW requires a ``siteconfig.py`` file tailored to Perlmutter.
This file must be present in the same directory where ``setup.py`` is run.
The directions below will use the ``siteconfig.py`` file stored here,

. After downloading the file you may customize it to your application.


To install, run the following commands::

  module load python cray-fftw
  conda create --name gpaw pip numpy scipy matplotlib
  source activate gpaw
  pip install ase
  git clone -b 23.9.1 https://gitlab.com/gpaw/gpaw.git
  cd gpaw
  wget https://raw.githubusercontent.com/NERSC/community-software/main/gpaw/siteconfig.py -O siteconfig.py
  python setup.py build_ext
  python setup.py install

**Note:** This will install GPAW version 23.9.1


Run
---

To run, you will first need to request an interactive session with the
``salloc`` command (see https://docs.nersc.gov/jobs/interactive/#interactive-jobs).
Once the session begins, try the commands::

  source activate gpaw
  export OMP_NUM_THREADS=1
  gpaw install-data test-datasets
  <when prompted to register the directory with `.../.gpaw/rc.py` select yes>
  srun -n 8 -c 2 gpaw test
  cat test.txt


Example ``siteconfig.py`` File
------------------------------

.. literalinclude:: customize_nersc_perlmutter.py
