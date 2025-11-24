.. _lrtddft2:

=========================================
Linear response TDDFT 2 - indexed version
=========================================

Ground state
============

The linear response TDDFT calculation needs a converged ground state
calculation with a set of unoccupied states.

We demonstrate the code usage for
:download:`(R)-methyloxirane molecule <lrtddft2/r-methyloxirane.xyz>`.

First, we calculate the ground state:

.. literalinclude:: lrtddft2/gs.py
    :start-after: Start

Text output: :download:`gs.out <lrtddft2/gs.out>`.

Then, we converge unoccupied states:

.. literalinclude:: lrtddft2/unocc.py
    :start-after: Start

Text output: :download:`unocc.out <lrtddft2/unocc.out>`.

Let's have a look on the states in the output file:

.. literalinclude:: lrtddft2/unocc.out
    :start-at: Band  Eigenvalues  Occupancy
    :end-before: Fermi level

We see that the Kohn-Sham eigenvalue difference between HOMO and
the highest converged unoccupied state is about 8.2 eV.
Thus, all Kohn-Sham single-particle transitions up to this energy
difference can be calculated from these unoccupied states.
If more is needed, then more unoccupied states would need to be converged.


.. note::

   Converging unoccupied states in some systems may
   require tuning the eigensolver.
   See the possible options in :ref:`the manual <manual_eigensolver>`.


Calculating response matrix and spectrum
========================================

The next step is to calculate the response matrix with
:class:`~gpaw.lrtddft2.LrTDDFT2`.

A very important convergence parameter is the number of Kohn-Sham
single-particle transitions used to calculate the response matrix.
This can be set through state indices
(see the parameters of :class:`~gpaw.lrtddft2.LrTDDFT2`),
or as demonstrated here,
through an energy cutoff parameter ``max_energy_diff``.
This parameter defines the maximum energy difference of
the Kohn-Sham transitions included in the calculation.

Note! If the used gpw file does not contain enough unoccupied states so
that **all** single-particle transitions defined by ``max_energy_diff``
can be included, then the calculation does not usually make sense.
Thus, check carefully the states in the unoccupied states calculation
(see the example above).

Note also! The ``max_energy_diff`` parameter does **not** mean that
the TDDFT excitations would be converged up to this energy.
Typically, the ``max_energy_diff`` needs to be much larger than the smallest
excitation energy of interest to obtain well converged results.
Checking the convergence with respect to the number of states
included in the calculation is crucial.

In this script, we set ``max_energy_diff`` to 7 eV.
We also show how to parallelize calculation
over Kohn-Sham electron-hole (eh) pairs with
:class:`~gpaw.lrtddft2.lr_communicators.LrCommunicators`
(8 tasks are used for each :class:`~gpaw.calculator.GPAW` calculator):

.. literalinclude:: lrtddft2/lr2.py
    :start-after: Start

Text output (:download:`lr2_with_07.00eV.out <lrtddft2/lr2_with_07.00eV.out>`)
shows the number of Kohn-Sham transitions within the set 7 eV limit:

.. literalinclude:: lrtddft2/lr2_with_07.00eV.out

The TDDFT excitations are
(:download:`transitions_with_07.00eV.dat <lrtddft2/transitions_with_07.00eV.dat>`):

.. literalinclude:: lrtddft2/transitions_with_07.00eV.dat


Restarting and recalculating
============================

The calculation can be restarted with the same script.
As an example, here we increase the energy cutoff to 8 eV.
The matrix elements calculated earlier up to 7 eV are reused,
and only the missing matrix elements are calculated:

.. literalinclude:: lrtddft2/lr2_restart.py
    :start-after: Start

Text output (:download:`lr2_with_08.00eV.out <lrtddft2/lr2_with_08.00eV.out>`)
shows the number of Kohn-Sham transitions within the set limit:

.. literalinclude:: lrtddft2/lr2_with_08.00eV.out

The TDDFT excitations are
(:download:`transitions_with_08.00eV.dat <lrtddft2/transitions_with_08.00eV.dat>`):

.. literalinclude:: lrtddft2/transitions_with_08.00eV.dat

It's important to note that also the first excitations change in comparison
to the earlier calculation with the 7 eV cutoff energy.
As stated earlier, the results must be converged with respect to
the cutoff energy, and typically the cutoff energy needs to be
much larger than the smallest excitation energy of interest.

Here we plot the photoabsorption and rotatory strength spectra from
the data files (
:download:`spectrum_with_07.00eV.dat <lrtddft2/spectrum_with_07.00eV.dat>` and
:download:`spectrum_with_08.00eV.dat <lrtddft2/spectrum_with_08.00eV.dat>`):

.. image:: lrtddft2/abs_spec.png

.. image:: lrtddft2/rot_spec.png

We note that the cutoff energy (and the number of unoccupied bands)
should be increased more to converge the spectra properly.


Analyzing spectrum
==================

Once the response matrix has been calculated, the same input script can be used
for calculating the spectrum and analyzing the transitions.
But, as all the expensive calculation is done already, it's sufficient
to run the script in serial.
Here is an example script for analysis without parallelization settings:

.. literalinclude:: lrtddft2/lr2_analyze.py
    :start-after: Start

The script produces the same spectra and transitions as above.
In addition, it demonstrates how to analyze transition contributions.
An example for the first TDDFT excitation
(:download:`tc_000_with_08.00eV.txt <lrtddft2/tc_000_with_08.00eV.txt>`):

.. literalinclude:: lrtddft2/tc_000_with_08.00eV.txt

Note that while this excitation is at about 5.9 eV, it has non-negligible
contributions from Kohn-Sham single-particle transitions above this energy.
This is generally the case, and it is basically the reason why
the ``max_energy_diff`` parameter has to be typically much higher
than the highest excitation energies of interest.


Quick reference
===============

.. autoclass:: gpaw.lrtddft2.LrTDDFT2
   :members:

.. autoclass:: gpaw.lrtddft2.lr_communicators.LrCommunicators
   :members:
