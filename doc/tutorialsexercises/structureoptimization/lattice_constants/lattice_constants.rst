.. _lattice_constants:

=========================
Finding lattice constants
=========================

.. seealso::

   * :mod:`ASE equation of state module <ase.eos>`
   * :ref:`ase:eos_example`
   * :ref:`ase:lattice_constant_example`


Fcc Aluminium
=============

Let's try to converge the lattice constant with respect to number of
plane-waves:

.. literalinclude:: al.py
    :end-before: al.calc.new

.. image:: Al_conv_ecut.png

Using a plane-wave cutoff energy of 400 eV, we now check convergence
with respect to number of **k**-points:

.. literalinclude:: al.py
    :start-after: al.get_potential_energy

.. image:: Al_conv_k.png

(see also :download:`analysis script <al_analysis.py>`).
