.. _rttddft:

==================================
New real-time TDDFT implementation
==================================

There is an ongoing effort in refactoring the real-time TDDFT codes.
There are two old implementations, one for :ref:`LCAO mode <lcao>`
and one for :ref:`FD mode <manual_stencils>`.
This page documents the new :class:`gpaw.new.rttddft.RTTDDFT` interface which is
common for both modes.

See :ref:`lcaotddft` and :ref:`timepropagation` for the old RT-TDDFT implementations.

:class:`gpaw.lcaotddft.dipolemomentwriter.DipoleMomentWriter`

Example usage
=============

First we do a standard ground-state calculation with the ``GPAW`` calculator
in :ref:`LCAO mode <lcao>`:

Some important points are:

* The grid spacing is only used to calculate the Hamiltonian matrix and
  therefore a coarser grid than usual can be used.
* Completely unoccupied bands should be left out of the calculation,
  since they are not needed.
* The density convergence criterion should be a few orders of magnitude
  more accurate than in usual ground-state calculations.
* One should use multipole-corrected Poisson solvers or
  other advanced Poisson solvers in any TDDFT run
  in order to guarantee the convergence of the potential with respect to
  the vacuum size.
  See the documentation on :ref:`advancedpoisson`.
  Currently, only the default Poisson solver works.
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

.. automodule:: gpaw.new.rttddft.td_algorithm
   :members:
