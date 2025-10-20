.. _lattice_constants:

=========================
Finding lattice constants
=========================

.. seealso::

   * `ASE EOS tutorial
     <https://ase-lib.org/tutorials/eos/eos.html>`_
   * `ASE Finding lattice constants tutorial
     <https://ase-lib.org/tutorials/lattice_constant.html>`_

   * `ASE equation of state module
     <https://ase-lib.org/ase/utils.html#equation-of-state>`_


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
