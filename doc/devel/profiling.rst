.. _profiling:

=========
Profiling
=========

cProfile
========

Python has a :mod:`cProfile` module to help you find the places in the
code where the time is spent.

Let's say you have a script ``script.py`` that you want to run through the
profiler.  This is what you do:

>>> import profile
>>> profile.run('import script', 'prof')

This will run your script and generate a profile in the file ``prof``.
You can also generate the profile by inserting a line like this in
your script::

    ...
    import cProfile
    cProfile.run('atoms.get_potential_energy()', 'prof')
    ...

.. note::

    Use::

        import cProfile
        from gpaw.mpi import rank
        cProfile.run('atoms.get_potential_energy()', f'prof-{rank:04}')

    if you want to run in parallel.

To analyse the results, you do this::

 >>> import pstats
 >>> pstats.Stats('prof').strip_dirs().sort_stats('time').print_stats(20)
 Tue Oct 14 19:08:54 2008    prof

         1093215 function calls (1091618 primitive calls) in 37.430 CPU seconds

   Ordered by: internal time
   List reduced from 1318 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    37074   10.310    0.000   10.310    0.000 :0(calculate_spinpaired)
     1659    4.780    0.003    4.780    0.003 :0(relax)
   167331    3.990    0.000    3.990    0.000 :0(dot)
     7559    3.440    0.000    3.440    0.000 :0(apply)
      370    2.730    0.007   17.090    0.046 xc_correction.py:130(calculate_energy_and_derivatives)
    37000    0.780    0.000    9.650    0.000 xc_functional.py:657(get_energy_and_potential_spinpaired)
    37074    0.720    0.000   12.990    0.000 xc_functional.py:346(calculate_spinpaired)
      ...
      ...

The list shows the 20 functions where the most time is spent.  Check
the :mod:`pstats` documentation if you want to do more fancy things.

.. tip::

   Since the :mod:`cProfile` module does not time calls to C-code, it
   is a good idea to run the code in debug mode - this will wrap
   calls to C-code in Python functions::

     $ python3 -d script.py

.. tip::

   There is also a quick and simple way to profile a script::

     $ python3 -m cProfile script.py


Parallel profiling (GPAW new only)
==================================

The profiling of GPAW new code is done by a decorator called trace, which
is to be applied to all functions one wants to profile.
GPAW already has a lot of trace decorators added.

To keep overhead minimum when not tracing,
An environment variable called ``GPAW_TRACE`` has to be set to 1
in order to allow tracing. If ``GPAW_TRACE`` is not defined, or 0,
the trace decorator will be identity, and now overhead will be added to function calls.

In addition to setting the environment variable,
one needs to use the ``global_timer`` of ``gpaw.new``. Below are two examples
of how to profile.


CPU profiling
-------------

.. literalinclude:: profiling.py

This will write ``.json`` files for each rank, and finally, at exit, it will concatenate them into a single file.
In order to visualize the pro

GPU profiling
-------------

.. literalinclude:: profiling.py


Adding tracing to new functions
-------------------------------
The trace decorator can be imported as follows: `from gpaw.new import trace`.

To profile a particular function one needs to decorate it as follows::

    from gpaw.new import trace
    ...
    @trace
    def my_slow_function('
    import cProfile
    cProfile.run('atoms.get_potential_energy()', 'prof')
    ...

