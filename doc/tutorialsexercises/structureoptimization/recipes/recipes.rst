.. _relax_recipes:

=================================
Structure relaxation recipes
=================================

In the previous sections we reviewed the basics of structure relaxations. 
Here, we give some examples how to run and refine structure optimizations with GPAW.

Fixed-cell relaxation
--------------------------------

The simplest scenario is to optimize the atom positions without changing
the unit cell shape or size. In this case, we need to calculate the forces 
acting on the atoms and pass these to an optimization routine (here: ``BFGS``)
which iteratively updates the atom positions such that the magnitude of 
the forces decreases jointly with the total potential energy.
First import the necessary modules:

.. literalinclude:: relax.py
    :end-before: literalinclude import-end

Now consider the function ``relax``:

* expects an ASE ``Atoms`` object, and 
* calculation parameters dictionary ``param``,
* optionally the maximum force ``fmax`` can be set,
* extracts calculation parameters from ``param`` (selected by the keyword ``fast_forces``),
* sets up a GPAW calculator,
* adds the initial magnetic moments, 
* optionally uses van der Waals forces with the DFT-D3 method, and 
* sets up the optimization.

.. literalinclude:: relax.py
    :start-after: literalinclude relax-start
    :end-before: literalinclude relax-end

The relaxation history is written to ``logname = 'opt.log'``, its trajectory is saved to
``trajname = 'opt.traj'`` and the final configuration is returned from the function.

The parameter file ``params_forces.py`` contains the calculation parameters in
standardized format.
Typically we would start with a fast and less accurate relaxation,
which is consecutively refined with tighter convergence parameters
once we approach the atomic configuration which minimizes the potential energy
(see :ref:`accuracy of the self-consistency cycle<manual_convergence>` 
and :ref:`converging forces<custom_convergence-forces>` for relevant options).

.. literalinclude:: params_forces.py

For more accurate force convergence the Brillioun zone sampling should be
refined by setting a larger number of kpoints (e.g. five or more points 
in each direction), and the convergences criteria should be tightend to
``"convergence": {"density": 1e-6, "forces": 1e-4}``.

Full relaxation
--------------------------------

The cell shape and size can be relaxed together with the atom positions using 
cell filters. For that, we need to import the corresponding class using 
``from ase.filters import FrechetCellFilter``.
The cell filter takes the ``Atoms`` object as input and is
directly handed to the optimization routine. For full relaxations, 
you can then simply replace the optimization configuration line 
``opt_conf = atoms`` with the ``opt_conf = FrechetCellFilter(atoms)``. 
In the example script we use an if-statement for that.

Note, that calculation of stresses requires the plane wave mode in GPAW
and that the accuracy of the calculations
should be refined (here: taking the more accurate parameter set ``params_stresses.py``).

.. literalinclude:: params_stresses.py

Here, you can download the full relaxation script :download:`relax.py`.
