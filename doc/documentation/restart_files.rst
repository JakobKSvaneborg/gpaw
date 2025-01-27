.. _restart_files:

=============
Restart files
=============

Writing restart files
=====================

Use ``calc.write('xyz.gpw')`` or ``calc.write('xyz.gpw', mode='all')``
to include also the wave functions:
:meth:`~gpaw.new.ase_interface.ASECalculator.write`.

.. tip::

   You can register an automatic call to the ``write`` method, every
   ``n``'th iteration of the SCF cycle like this::

     calc.attach(calc.write, n, 'xyz.gpw')

   or::

     calc.attach(calc.write, n, 'xyz.gpw', mode='all')

   This can be useful for very expensive calculations, where the SCF cycle
   may be interrupted before it completes. In this way, you can resume the
   calculation from an intermediate electronic structure.


Reading restart files
=====================

The calculation can be read from file like this::

  calc = GPAW('xyz.gpw')

or this::

  atoms, calc = restart('xyz.gpw')
