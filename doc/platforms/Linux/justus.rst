=======
justus2
=======

Information about `justus2 <https://wiki.bwhpc.de/e/Category:BwForCluster_JUSTUS_2>`__.

Building GPAW
=============

We assume that the installation will be located in ``$SOURCEDIR``, which might be set to::

  export SOURCEDIR=$HOME/source

for example.

Setups
------

The setups of your choice must be installed
(see also :ref:`installation of paw datasets`)::

  cd
  GPAW_SETUP_SOURCE=$SOURCEDIR/gpaw-setups
  mkdir -p $GPAW_SETUP_SOURCE
  cd $GPAW_SETUP_SOURCE
  wget https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.11271.tar.gz
  tar xzf gpaw-setups-0.9.11271.tar.gz

Let gpaw know about the setups::

  export GPAW_SETUP_PATH=$GPAW_SETUP_SOURCE/gpaw-setups-0.9.11271

Using the module environment
----------------------------

It is very handy to add our installation to the module environment::

  cd
  mkdir -p modulefiles/gpaw-setups
  cd modulefiles/gpaw-setups
  echo -e "#%Module1.0\nprepend-path       GPAW_SETUP_PATH    $GPAW_SETUP_SOURCE/gpaw-setups-0.9.11271" > 0.9.11271

We need to let the system know about our modules
(add this command to ``~/.profile`` or ``~/.bashrc`` to execute automatically)::

  module use $HOME/modulefiles

such that we also see them with::

  module avail

libxc
-----

GPAW relies on libxc (see the `libxc web site <http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:download>`__). 
To install libxc we assume that ``MYLIBXCDIR`` is set to 
the directory where you want to install 
(e.g. ``MYLIBXCDIR=$SOURCEDIR/libxc``)::

 mkdir -p $MYLIBXCDIR
 cd $MYLIBXCDIR
 wget -O libxc-5.0.0.tar.gz http://www.tddft.org/programs/libxc/down.php?file=5.0.0/libxc-5.0.0.tar.gz
 tar xzf libxc-5.0.0.tar.gz
 cd libxc-5.0.0
 mkdir install
 ./configure CFLAGS="-fPIC" --prefix=$PWD/install -enable-shared
 make |tee make.log
 make install

This will have installed the libs ``$MYLIBXCDIR/libxc-5.0.0/install/lib`` 
and the C header
files to ``$MYLIBXCDIR/libxc-5.0.0/install/include``.
We create a module for libxc::

  cd
  mkdir modulefiles/libxc
  cd modulefiles/libxc

and edit the module file  :file:`5.0.0` that should read::

  #%Module1.0

  #                                    change this to your path
  set             libxchome            /home/fr/fr_fr/fr_mw767/source/libxc/libxc-5.0.0/install
  prepend-path    C_INCLUDE_PATH       $libxchome/include
  prepend-path    LIBRARY_PATH         $libxchome/lib
  prepend-path    LD_LIBRARY_PATH      $libxchome/lib

ASE release
-----------

You might want to install a stable version of ASE::

  cd
  ASE_SOURCE=$PWD/source/ase
  mkdir -p $ASE_SOURCE
  cd $ASE_SOURCE
  git clone -b 3.18.1 https://gitlab.com/ase/ase.git 3.18.1

We add our installation to the module environment::

  cd
  mkdir -p modulefiles/ase
  cd modulefiles/ase
  
Edit the module file  :file:`3.18.1` that should read::

  #%Module1.0

  if {![is-loaded numlib/python_scipy]} {module load numlib/python_scipy/1.1.0-python_numpy-1.14.0-python-3.5.0}

  #           change this to your path
  set asehome /home/fr/fr_fr/fr_mw767/source/ase/3.18.1
  prepend-path       PYTHONPATH    $asehome
  prepend-path       PATH          $asehome/tools

ASE trunk
---------

We get ASE trunk::

  cd
  ASE_SOURCE=$PWD/source/ase
  mkdir -p $ASE_SOURCE
  cd $ASE_SOURCE
  git clone https://gitlab.com/ase/ase.git trunk

which can be updated using::

  cd $ASE_SOURCE/trunk
  git pull

We add our installation to the module environment::

  cd
  mkdir -p modulefiles/ase
  cd modulefiles/ase

and edit the module file  :file:`trunk` that should read::

  #%Module1.0

  if {![is-loaded numlib/python_scipy]} {module load numlib/python_scipy}

  #           change this to your path
  set asehome /home/fr/fr_fr/fr_mw767/source/ase/trunk
  prepend-path       PYTHONPATH    $asehome
  prepend-path       PATH          $asehome/tools

matplotlib
----------

In order to use `ase gui` in it's full strength it is useful to install
`matplotlib` via pip::

  python3 -m pip install matplotlib

Building GPAW
-------------

We create a place for gpaw and get it::

 cd $SOURCEDIR
 git clone https://gitlab.com/gpaw/gpaw.git

The current version can then be updated by::

 cd $SOURCEDIR/gpaw
 git pull

A specific tag can be loaded by::

 cd $GPAW_SOURCE/trunk
 # list tags
 git tag
 # load version 1.2.0
 git checkout 1.2.0

To build the current trunk version of GPAW we need to create
a file :file:`siteconfig.py` that reads

.. literalinclude:: nemo_siteconfig.py

To build GPAW use::

 module purge
 module load libxc
 module load ase
 module load compiler/intel
 module load mpi/impi

 cd $GPAW_SOURCE/trunk
 CC=mpicc; python3 setup.py build

which builds GPAW to ``$GPAW_SOURCE/trunk/build``.
We create a module that creates the necessary definitions::

  cd
  mkdir -p modulefiles/gpaw
  cd modulefiles/gpaw

The file  :file:`master` that should read::

 #%Module1.0

 if {![is-loaded ase]} {module load ase}
 if {![is-loaded libxc]} {module load libxc}
 if {![is-loaded mpi]}  {module load mpi/impi}
 if {![is-loaded compiler/intel]} {module load compiler/intel}
 if {![is-loaded gpaw-setups]}  {module load gpaw-setups}

 # change the following directory definition to your needs
 set gpawhome /home/fr/fr_fr/fr_mw767/source/gpaw
 # this can stay as is
 prepend-path    PATH                 $gpawhome/tools:$gpawhome/build/scripts-3.8
 prepend-path    PYTHONPATH           $gpawhome:$gpawhome/build/lib.linux-x86_64-3.8
 

Running GPAW
------------

A gpaw script :file:`test.py` can be submitted with the help
of :file:`gpaw-runscript` to run on 48 cores like this::

  > module load gpaw
  > gpaw-runscript test.py 48
  using justus2
  run.justus2 written
  > sbatch run.justus
