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
The following example script:

* reads in the structure from ``unrelaxed.json``, 
* reads the calculator parameters from ``params.json`` (selected by the keyword ``fast_forces``),
* sets up a GPAW calculator,
* adds the initial magnetic moments, 
* optionally uses van der Waals forces with the DFT-D3 method, and 
* runs the optimization. 

The relaxation history is written to ``opt.log``, its trajectory is saved to
``opt.traj`` and the final configuration is written to ``relaxed.json``.

.. literalinclude:: fixcell_relax.py

The parameter file ``params.json`` contains the calculation parameters in
standardized format.
Typically we would start with a fast and less accurate relaxation,
which is consecutively refined with tighter convergence parameters
once we approach the atomic configuration which minimizes the potential energy
(see :ref:`accuracy of the self-consistency cycle<manual_convergence>` 
and :ref:`converging forces<custom_convergence-forces>` for relevant options).

.. literalinclude:: params_forces.json

For more accurate force convergence the Brillioun zone sampling should be
refined by setting a larger number of kpoints (e.g. five or more points 
in each direction), and the convergences criteria should be tightend to
``"convergence": {"density": 1e-6, "forces": 1e-4}``.

Full relaxation
--------------------------------

The cell shape and size can be relaxed together with the atom positions using 
cell filters. For that, we need to import the corresponding class using 
``from ase.filters import UnitCellFilter``.
The cell filter takes the ``Atoms`` object as input and is
directly handed to the optimization routine. For full relaxations, 
you can then simply replace the  optimization routine line 
``opt = ...`` with the following code lines

.. literalinclude:: full_relax.py
    :start-after: literalinclude start line
    :end-before: literalinclude end line

Note, that calculation of stresses requires the plane wave mode in GPAW
and that the accuracy of the calculations
should be refined (here: taking the more accurate parameter set by
using the parameter keyword ``param_key = accurate_stresses``).

.. literalinclude:: params_stresses.json

Parameter files using multi-level dictionaries (with keys such as 
``fast_forces``, ``accurate_stresses``) 
allow to store different parameter sets in one common parameter file. 
The appropriate parameters are selected by keywords in the
optimization script. 

Here, you can download an example structure :download:`unrelaxed.json` (bulk hBN), 
the relaxation scripts :download:`fixcell_relax.py`, :download:`full_relax.py` and 
the joint parameter file :download:`params.json`.
