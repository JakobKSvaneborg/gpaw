Cohesive energy of bulk FCC Pt
==============================

When calculating cohesive energies, one will need to calculate the energy
of isolated atoms.  For some atoms it can be difficult to converge to the
correct magnetic state.  For those difficult cases it often helps to use
:ref:`directmin` instead of the default Davidson eigensolver.  Here is
an example for Pt:

.. literalinclude::  pt.py

One should take a careful look at the ``pt-atom.txt`` to check that one has
found the correct state.  Here is a more thorough analysis:

.. literalinclude::  projections.py

This table shows the orbital characters for majority and minority spins:

.. csv-table::
   :file: pt-atom.csv
