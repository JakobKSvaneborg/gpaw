.. _restart_files:

=============
Restart files
=============

Writing restart files
=====================

It is possible to save a ground state by writing a .gpw file.
This saves the density, potential, eigenvalues and other information.


Use the :meth:`~gpaw.new.ase_interface.ASECalculator.write` method.
For example ``calc.write('xyz.gpw')``.

 * To also save the (potentially very large) wavefunctions, use
   ``calc.write('xyz.gpw', mode='all')``.

With :ref:`newgpaw` there are also two ways to produce smaller .gpw files:

 * To avoid saving the (potentially large) PAW projections, use
   ``calc.write('xyz.gpw', include_projection=False)``.
 * To further save space you can save the file in single-precision
   representation
   using ``calc.write('xyz.gpw', precision='single')``. Note that
   reloading the file will load it as double-precision, so
   further processing will always be double precision.


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
