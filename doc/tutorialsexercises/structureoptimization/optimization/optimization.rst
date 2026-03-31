.. _structure_optimization:

========================
 Structure optimization
========================

In this tutorial we consider structure optimization of the :mol:`H_2` molecule. For that, we will calculate the atomization energy of the molecule both for the experimentally determined geometry and for the structure relaxed using the GPAW calculator.

In a previous tutorial we have calculated atomization energies. As a short recap, the following script will calculate the atomization energy of a hydrogen molecule with the geometry from experiment:

.. literalinclude:: atomize.py

Above, we calculated the atomization energy for
:mol:`H_2` using the experimental bond length of 0.74 Å.  In
this tutorial, we ask an :mod:`ASE optimizer <ase.optimize>`
to iteratively find
the structural energy minimum, where all atomic forces are below 0.05
eV/Å.  The following script will do the job:

.. literalinclude:: relax.py

The result is:

.. literalinclude:: optimization.txt

To save time you could have told the minimizer to keep one atom fixed,
and only relaxing the other. This is achieved through the use of
constraints::

  molecule.set_constraint(FixAtoms(mask=[0, 1]))

The keyword ``mask`` contains list of booleans for each atom indicating
whether the atom's position should be fixed or not. See the
:mod:`ase.constraints` module on the ASE page for
more information and examples for setting constraints.
