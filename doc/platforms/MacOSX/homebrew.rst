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

Modern MacOS installations no longer permits pip installation with
``--user``, neither with the system Python nor with the Homebrew
Python.  You therefore have to create a virtual environment, and
install GPAW in it.

Create and activate the virtual environment:

    $ python3 -m venv venv_gpaw
    $ source venv_gpaw/bin/activate

Update pip::

    $ pip install --upgrade pip

Install ASE::

    $ pip install --upgrade ase

Use this :ref:`siteconfig.py <siteconfig>` file, download it and place
it in the folder ``~/.gpaw``, or set the environment variable
``$GPAW_CONFIG`` to the full path and name of the file (placing the
file in current working directory no longer works with a modern Python):

.. literalinclude:: siteconfig.py

Install GPAW::

    $ pip install --upgrade gpaw

Test GPAW::

    $ gpaw test
    $ gpaw -P 4 test

Check the output to see that it was compiled with MPI, scalapack and
FFTW.  If not, it failed to find the ``siteconfig.py`` file.
