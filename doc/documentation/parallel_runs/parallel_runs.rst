.. _parallel_runs:

=============
Parallel runs
=============

.. toctree::
   :maxdepth: 1

.. _parallel_running_jobs:


Running jobs in parallel
========================

Parallel calculations are done primarily with MPI.
The parallelization can be done over the **k**-points, bands,
and using real-space domain decomposition.
The code will try to make a sensible domain
decomposition that match both the number of processors and the size of
the unit cell.  This choice can be overruled, see
:ref:`manual_parallelization_types`. Complementary OpenMP
parallelization can improve the performance in some cases, see
:ref:`manual_openmp`.

Before starting a parallel calculation, it might be useful to check how the
parallelization corresponding to the given number of processes would be done
with the ``--dry-run=N`` command line option::

    $ gpaw python --dry-run=8 script.py

The output will contain also the "Calculator" RAM Memory estimate per process.

In order to run GPAW in parallel, you
do one of these two::

    $ mpiexec -n <cores> gpaw python script.py
    $ gpaw -P <cores> python script.py
    $ mpiexec -n <cores> python3 script.py

The first two are the recommended ones:  The *gpaw* script will make sure
that imports are done in an efficient way.

.. tip::

   You can use the :envvar:`GPAW_MPI_OPTIONS` to pass options to ``mpiexex``.
   Example::

     GPAW_MPI_OPTIONS="--oversubscribe"

.. envvar:: GPAW_MPI_OPTIONS

    Options for ``mpiexec``.


Submitting a job to a queuing system
====================================

You can write a shell-script that contains this line::

    mpiexec gpaw python script.py

and then submit that with ``sbatch``, ``qsub`` or some other command.

Alternatives:

* If you are on a SLURM system:  use the :ref:`sbatch <cli>` sub-command
  of the ``gpaw`` command-line tool::

      $ gpaw sbatch -- [sbatch options] script.py [script options]

* Use MyQueue_::

      $ mq submit "script.py [script options]" -R <resources>

* Write you own *submit* script.  See this example:
  :git:`doc/platforms/gbar/qsub.py`.

.. _MyQueue: https://myqueue.readthedocs.io/


Alternative submit tool
=======================

Alternatively, the script gpaw-runscript can be used, try::

  $ gpaw-runscript -h

to get the architectures implemented and the available options. As an
example, use::

  $ gpaw-runscript script.py 32

to write a job submission script running script.py on 32 cpus.
The tool tries to guess the architecture/host automatically.

By default it uses the following environment variables to write the runscript:

=============== ===================================
variable        meaning
=============== ===================================
HOSTNAME        name used to assign host type
PYTHONPATH      path for Python
GPAW_SETUP_PATH where to find the setups
GPAW_MAIL       where to send emails about the jobs
=============== ===================================


Writing to files
================

Be careful when writing to files in a parallel run.  Instead of ``f =
open('data', 'w')``, use:

>>> from ase.parallel import paropen
>>> f = paropen('data', 'w')

Using ``paropen``, you get a real file object on the master node, and dummy
objects on the slaves.  It is equivalent to this:

>>> from ase.parallel import world
>>> if world.rank == 0:
...     f = open('data', 'w')
... else:
...     f = open('/dev/null', 'w')

If you *really* want all nodes to write something to files, you should make
sure that the files have different names:

>>> from ase.parallel import world
>>> f = open('data.{}'.format(world.rank), 'w')


Writing text output
===================

Text output written by the ``print`` statement is written by all nodes.
To avoid this use:

>>> from ase.parallel import parprint
>>> print('This is written by all nodes')
>>> parprint('This is written by the master only')

which is equivalent to

>>> from ase.parallel import world
>>> print('This is written by all nodes')
>>> if world.rank == 0:
...     print('This is written by the master only')


.. _different_calculations_in parallel:

Running different calculations in parallel
==========================================

A GPAW calculator object will per default distribute its work on all
available processes. If you want to use several different calculators
at the same time, however, you can specify a set of processes to be
used by each calculator. The processes are supplied to the constructor,
either by specifying an :ref:`MPI Communicator object <communicators>`,
or simply a list of ranks. Thus, you may write::

  from gpaw import GPAW
  import gpaw.mpi as mpi

  # Create a calculator using ranks 0, 3 and 4 from the mpi world communicator
  ranks = [0, 3, 4]
  comm = mpi.world.new_communicator(ranks)
  if mpi.world.rank in ranks:
      calc = GPAW(communicator=comm, ...)
      ...

Be sure to specify different output files to each calculator,
otherwise their outputs will be mixed up.

Here is an example which calculates the atomization energy of a
nitrogen molecule using two processes:

.. literalinclude:: parallel_atomization.py


.. _manual_parallelization_types:
.. _manual_parallel:

Parallelization options
=======================

In version 0.7, a new keyword called ``parallel`` was introduced to provide
a unified way of specifying parallelization-related options. Similar to
the way we :ref:`specify convergence criteria <manual_convergence>` with the
``convergence`` keyword, a Python dictionary is used to contain all such
options in a single keyword.

The default value corresponds to this Python dictionary::

  {'kpt':                 None,
   'domain':              None,
   'band':                None,
   'order':               'kdb',
   'stridebands':         False,
   'augment_grids':       False,
   'sl_auto':             False,
   'sl_default':          None,
   'sl_diagonalize':      None,
   'sl_inverse_cholesky': None,
   'sl_lcao':             None,
   'sl_lrtddft':          None,
   'use_elpa':            False,
   'elpasolver':          '2stage',
   'buffer_size':         None}

In words:

* ``'kpt'`` is an integer and denotes the number of groups of k-points over
  which to parallelize.  k-point parallelization is the most efficient type of
  parallelization for most systems with many electrons and/or many k-points. If
  unspecified, the calculator will choose a parallelization itself which
  maximizes the k-point parallelization unless that leads to load imbalance; in
  that case, it may prioritize domain decomposition.
  Note: parallelization over spin is not possible in
  :ref:`GPAW 20.10.0 and newer versions <releasenotes>`.

* The ``'domain'`` value specifies either an integer ``n`` or a tuple
  ``(nx,ny,nz)`` of 3 integers for
  :ref:`domain decomposition <manual_parsize_domain>`.
  If not specified (i.e. ``None``), the calculator will try to determine the
  best domain parallelization size based on number of kpoints etc.

* The ``'band'`` value specifies the number of parallelization groups to use
  for :ref:`band parallelization <manual_parsize_bands>`. If not specified (i.e. ``None``), the calculator will try to determine the best band parallelization size based on number of kpoints etc.

* ``'order'`` specifies how different parallelization modes are nested
  within the calculator's world communicator.  Must be a permutation
  of the characters ``'kdb'`` which is the default.  The characters
  denote k-point, domain or band parallelization respectively.  The
  last mode will be assigned contiguous ranks and thus, depending on
  network layout, probably becomes more efficient.  Usually for static
  calculations the most efficient order is ``'kdb'`` whereas for TDDFT
  it is ``'kbd'``.

* The ``'stridebands'`` value only applies when band parallelization is used,
  and can be used to toggle between grouped and strided band distribution.

* If ``'augment_grids'`` is ``True``, all cores will be used for XC/Poisson solver. When parallelizing over k-points or bands, in the planewave mode, and using ScaLAPACK, setting ``'augment_grids'`` to True will make use of all cores including those for k-point and band parallelization.

* If ``'sl_auto'`` is ``True``, ScaLAPACK will be enabled with automatically
  chosen parameters and using all available CPUs.

* The other ``'sl_...'`` values are for using ScaLAPACK with different
  parameters in different operations.
  Each can be specified as a tuple ``(m,n,mb)`` of 3 integers to
  indicate an ``m*n`` grid of CPUs and a block size of ``mb``.
  If any of the three latter keywords are not
  specified (i.e. ``None``), they default to the value of
  ``'sl_default'``. Presently, ``'sl_inverse_cholesky'`` must equal
  ``'sl_diagonalize'``.

* If the Elpa library is installed, enable it by setting ``use_elpa``
  to ``True``.  Elpa will be used to diagonalize the Hamiltonian.  The
  Elpa distribution relies on BLACS and ScaLAPACK, and hence can only
  be used alongside ``sl_auto``, ``sl_default``, or a similar keyword.
  Enabling Elpa is highly recommended as it significantly
  speeds up the diagonalization step.  See also :ref:`lcao`.

* ``elpasolver`` indicates which solver to use with Elpa.  By default
  it uses the two-stage solver, ``'2stage'``.  The other allowed value
  is ``'1stage'``.  This setting will only have effect if Elpa is enabled.

* The ``'buffer_size'``  is specified as an integer and corresponds to
  the size of the buffer in KiB used in the 1D systolic parallel
  matrix multiply algorithm. The default value corresponds to sending all
  wavefunctions simultaneously. A reasonable value would be the size
  of the largest cache (L2 or L3) divide by the number of MPI tasks
  per CPU. Values larger than the default value are non-sensical and
  internally reset to the default value.

.. note::
   With the exception of ``'stridebands'``, these parameters all have an
   equivalent command line argument which can equally well be used to specify
   these parallelization options. Note however that the values explicitly given
   in the ``parallel`` keyword to a calculator will override those given via
   the command line. As such, the command line arguments thus merely redefine
   the default values which are used in case the ``parallel`` keyword doesn't
   specifically state otherwise.


.. _manual_parsize_domain:

Domain decomposition
--------------------

Any choice for the domain decomposition can be forced by specifying
``domain`` in the ``parallel`` keyword. It can be given in the form
``parallel={'domain': (nx,ny,nz)}`` to force the decomposition into ``nx``,
``ny``, and ``nz`` boxes in x, y, and z direction respectively. Alternatively,
one may just specify the total number of domains to decompose into, leaving
it to an internal cost-minimizer algorithm to determine the number of domains
in the x, y and z directions such that parallel efficiency is optimal. This
is achieved by giving the ``domain`` argument as ``parallel={'domain': n}``
where ``n`` is the total number of boxes.

.. tip::
   ``parallel={'domain': world.size}`` will force all parallelization to be
   carried out solely in terms of domain decomposition, and will in general
   be much more efficient than e.g. ``parallel={'domain': (1,1,world.size)}``.
   You might have to add ``from gpaw.mpi import world`` to the script to
   define ``world``.


.. _manual_parsize_bands:

Band parallelization
--------------------

Parallelization over Kohn-Sham orbitals (i.e. bands) becomes favorable when
the number of bands `N` is so large that `\mathcal{O}(N^2)`
operations begin to dominate in terms of computational time. Linear algebra
for orthonormalization and diagonalization of the wavefunctions is the most
noticeable contributor in this regime, and therefore, band parallelization
can be used to distribute the computational load over several CPUs. This
is achieved by giving the ``band`` argument as ``parallel={'band': nbg}``
where ``nbg`` is the number of band groups to parallelize over.

.. tip::
   Whereas band parallelization in itself will reduce the amount of operations
   each CPU has to carry out to calculate e.g. the overlap matrix, the actual
   linear algebra necessary to solve such linear systems is in fact still
   done using serial LAPACK by default. It is therefor advisable to use both
   band parallelization and ScaLAPACK in conjunction to reduce this
   potential bottleneck.

More information about these topics can be found here:

.. toctree::
   :maxdepth: 1

   band_parallelization/band_parallelization


.. _manual_ScaLAPACK:

ScaLAPACK
---------

ScaLAPACK improves performance of calculations beyond a certain size.
This size depends on whether using FD, LCAO, or PW mode.

In FD or PW mode, ScaLAPACK operations are applied to arrays of size
nbands by nbands, whereas in LCAO mode, the arrays are generally the
number of orbitals by the number of orbitals and therefore larger,
making ScaLAPACK particularly important for LCAO calculations.

With LCAO, it starts to become an advantage to use ScaLAPACK at around
800 orbitals which corresponds to about 50 normal (non-hydrogen,
non-semicore) atoms with standard DZP basis set.
In FD mode, calculations with nbands > 500 will
benefit from ScaLAPACK; otherwise, the default serial LAPACK might as
well be used.

The ScaLAPACK parameters
are defined using the parallel
keyword dictionary, e.g., ``sl_default=(m, n, block)``.

A block size of 64 has been found to be a universally good choice both
in all modes.

In LCAO mode, it is normally best to assign as many cores as possible,
which means that ``m`` and ``n`` should multiply to the total number of cores
divided by the k-point parallelization.
For example with 128 cores and parallelizing by 4 over k-points,
there are 32 cores per k-point available per scalapack and a sensible
choice is ``m=8``, ``n=4``.  You can use ``sl_auto=True`` to make
such a choice automatically.

In FD or PW mode, a good guess for these
parameters on most systems is related to the numbers of bands.
We recommend for FD/PW::

  mb = 64
  m = floor(sqrt(nbands/mb))
  n = m

There are a total of four ``'sl_...'`` keywords. Most people will be
fine just using ``'sl_default'`` or even ``'sl_auto'``. Here we use the same
ScaLAPACK parameters in three different places: i) general eigensolve
in the LCAO initialization ii) standard eigensolve in the FD calculation and
iii) Cholesky decomposition in the FD calculation. It is currently
possible to use different ScaLAPACK parameters in the LCAO
initialization and the FD calculation by using two of the ScaLAPACK
keywords in tandem, e.g::

  GPAW(..., parallel={'sl_lcao': (p, q, p), 'sl_default': (m, n, mb)})

where ``p``, ``q``, ``pb``, ``m``, ``n``, and ``mb`` all
have different values. The most general case is the combination
of three ScaLAPACK keywords.
Note that some combinations of keywords may not be supported.


.. _manual_openmp:

Hybrid OpenMP/MPI parallelization
---------------------------------

In some hardware the performance of large FD and LCAO and calculations
can be improved by using OpenMP parallelization in addition to
MPI. When GPAW is built with OpenMP support, hybrid parallelization
is enabled by setting the OMP_NUM_THREADS environment variable::

  export OMP_NUM_THREADS=4
  mpiexec -n 512 gpaw python script.py

This would run the calculation with a total of 2048 CPU cores. As the
optimum MPI task / OpenMP thread ratio depends a lot on the particular
input and underlying hardware, it is recommended to experiment with
different settings before production calculations.
