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

Install pip::

    $ sudo easy_install pip

Install required Python packages::

    $ pip install numpy scipy matplotlib

Install and test ASE::

    $ pip install --upgrade --user ase
    $ python -m ase test

Use this :ref:`siteconfig.py <siteconfig>` file:

.. literalinclude:: siteconfig.py

Install GPAW::

    $ pip install --upgrade --user gpaw

Test GPAW::

    $ gpaw test
    $ gpaw -P 4 test
