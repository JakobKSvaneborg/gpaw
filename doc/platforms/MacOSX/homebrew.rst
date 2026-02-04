.. _homebrew:

========
Homebrew
========

.. highlight:: bash

Get Xcode from the App Store and install it. You also need to install the
command line tools, do this with the command::

    $ xcode-select --install

Install Homebrew::

    $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    $ echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bash_profile

Install ASE and GPAW dependencies::

    $ brew install python
    $ brew install gcc
    $ brew install libxc
    $ brew install open-mpi
    $ brew install fftw
    $ brew install openblas
    $ brew install scalapack

Modern MacOS installations no longer permits pip installation with
``--user``, neither with the system Python nor with the Homebrew
Python.  You therefore have to create a virtual environment, and
install GPAW in it.

Create and activate the virtual environment::

    $ python3 -m venv venv_gpaw
    $ source venv_gpaw/bin/activate

Update pip::

    $ pip install --upgrade pip

Install ASE::

    $ pip install --upgrade ase

You need to create a ``siteconfig.py`` file and place it in the folder ``~/.gpaw``, or set the environment variable
``$GPAW_CONFIG`` to the full path and name of the file (placing the
file in current working directory no longer works with a modern
Python).  You can :ref:`read more about siteconfig.py here
<siteconfig>`.  Unfortunately, the file needs to be different if your Mac has an Apple
Silicon or an Intel chip.

Use this :download:`siteconfig.py` file if you has an Apple Silicon
processor (a reasonably new Mac):

.. literalinclude:: siteconfig.py

Use this :download:`siteconfig.py <siteconfig_intel.py>` file if you has an Intel
processor:

.. literalinclude:: siteconfig_intel.py

Install GPAW::

    $ pip install --no-cache-dir --upgrade gpaw

(the ``--no-cache-dir`` option makes sure that GPAW is recompiled if
you have fixed something in ``siteconfig.py``)

Test GPAW::

    $ gpaw test
    $ gpaw -P 4 test

Check the output to see that it was compiled with MPI, scalapack and
FFTW.  If not, it failed to find the ``siteconfig.py`` file.
