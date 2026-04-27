.. _rttddft:

==================================
New real-time TDDFT implementation
==================================

There is an ongoing effort in refactoring the real-time TDDFT codes
to comply with the :ref:`new GPAW <newgpaw>` backend.
There are two old rt-TDDFT implementations, one for :ref:`LCAO mode <lcao>`
and one for :ref:`FD mode <manual_stencils>`.
This page documents the new :class:`gpaw.new.rttddft.RTTDDFT` interface which is
common for both modes.
See :ref:`lcaotddft` and :ref:`timepropagation` for the old rt-TDDFT implementations.

Ported features
---------------
The following features from the old code have been ported

* Reading and writing of restart files
* Delta-kicks
* The Explicit Crank-Nicolson and Semi-implicit Crank-Nicolson propagators

The following features are still to be ported

* Logging
* Parallel execution
* Time-dependent potentials
* The radiation-reaction potential
* Linearization of the xc functional
* Scaling factor for the dynamic part of the hamiltonian


User interface
--------------
The new interface can be used directly. The :class:`gpaw.new.rttddft.RTTDDFT`
can be initialized from

* A DFT calculation (:func:`~gpaw.new.rttddft.RTTDDFT.from_dft_calculation`)
* A DFT file (:func:`~gpaw.new.rttddft.RTTDDFT.from_dft_file`)
* A rt-TDDFT restart file (:func:`~gpaw.new.rttddft.RTTDDFT.from_rttddft_file`)

When starting from a DFT calculation or DFT file, the propagation algorithm
can be specified using the :code:`td_algorithm` keyword.

* :code:`ecn` for the :class:`gpaw.new.rttddft.td_algorithm.ECNAlgorithm`.
* :code:`sicn` for the :class:`gpaw.new.rttddft.td_algorithm.SICNAlgorithm`.

Example
-------

First we do a standard ground-state calculation with the ``GPAW`` calculator
in :ref:`LCAO mode <lcao>`:

Some important points are:

* In :ref:`LCAO mode <lcao>`, the grid spacing is only used to calculate
  the Hamiltonian matrix and therefore a coarser grid than usual can be used.
* Completely unoccupied bands should be left out of the calculation,
  since they are not needed.
* The density convergence criterion should be a few orders of magnitude
  more accurate than in usual ground-state calculations.
* One should use multipole-corrected Poisson solvers or
  other advanced Poisson solvers in any TDDFT run
  in order to guarantee the convergence of the potential with respect to
  the vacuum size.
  See the documentation on :ref:`advancedpoisson`.
* Point group symmetries are disabled in TDDFT, since the symmetry is
  broken by the time-dependent potential.
  The TDDFT calculation will refuse to start if the ground state
  has been converged with point group symmetries enabled.

.. literalinclude:: rttddft.py
   :start-after: P1
   :end-before: P2


Next we kick the system in the z direction and propagate 3000 steps of 0.001 ASE units.
The ASE unit of time is `Å\sqrt{\mathrm{e/eV}}`, which is about 10.18fs.
We open a file and write to it the dipole moments.

* It is important to explicitly write the kick to the dipole moment file so that
  the spectrum calculator knows what kind of kick was used.
  The convenience property `most_recent_kick` of the history object can be used to
  obtain the last kick.

.. literalinclude:: rttddft.py
   :start-after: P2
   :end-before: P3

After the time propagation, the spectrum can be calculated:

.. literalinclude:: rttddft.py
   :start-after: P3

This example input script can be downloaded :download:`here <rttddft.py>`.


Code documentation
==================

.. automodule:: gpaw.new.rttddft
   :members:

.. automodule:: gpaw.new.rttddft.history
   :members:

.. class:: TDAlgorithmLike

   Instance of :class:`gpaw.new.rttddft.td_algorithm.TDAlgorithm` or
   a string or dict that describes the :class:`gpaw.new.rttddft.td_algorithm.TDAlgorithm`.

   Allowed strings are

   * `ecn` for the :class:`gpaw.new.rttddft.td_algorithm.ECNAlgorithm`.
   * `sicn` for the :class:`gpaw.new.rttddft.td_algorithm.SICNAlgorithm`.

   Allowed dictionaries are on the form :code:`{'name': name, ...}`
   where `name` is one of the allowed strings.

.. automodule:: gpaw.new.rttddft.td_algorithm
   :members:
   :undoc-members:

.. automodule:: gpaw.new.rttddft.dataclasses
   :members:

.. autoclass:: gpaw.external.ExternalPotential
