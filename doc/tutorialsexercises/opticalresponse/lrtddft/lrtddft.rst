.. _exercise_lrtddft:

=========================================
Calculation of optical spectra with TDDFT
=========================================

In this exercise we calculate optical spectrum of Na2 molecule using
linear response time-dependent density functional theory (see also
:ref:`lrtddft`). We start with a normal ground state calculation:

.. literalinclude:: Na2TDDFT.py
.. highlight:: python

Once the ground state calculation with unoccupied states is finished, the last
part of the script performs a linear response TDDFT calculation.

As the construction of the Omega matrix is computationally the most intensive
part it is sometimes convenient to perform diagonalization and construction of
spectrum in separate calculations:

.. literalinclude:: part2.py
.. highlight:: python

The number of electron-hole pairs used in the calculation can be controlled with
``istart`` and ``jend`` options of LrTDDFT::

  LrTDDFT(calc, restrict={'istart':0, 'jend': 10})

By default only singlet-singlet transitions are calculated, singlet-triplet
transitions can be calculated by giving the ``nspins`` parameter::

  LrTDDFT(calc, restrict={'istart': 0, 'jend': 10}, nspins=2)


1. Check how the results vary with the number of unoccupied states in
   the calculation (``jend`` parameter).

2. Calculate also singlet-triplet transitions. Why do they not show up
   in the spectrum?

3. Check how the results vary with the empty space around the molecule.

4. Try to calculate optical spectrum also with the
   :ref:`timepropagation` approach and see how the results compare to
   linear response calculation.  Note that the :ref:`timepropagation`
   examples deal with Be2, you can of course modify it to use Na2 instead.
