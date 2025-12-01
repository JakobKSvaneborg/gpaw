.. _ps2ae:

Obtaining all-electron wave functions and electrostatic potential
=================================================================

Wave functions
--------------

To get from the pseudo (PS) wave function to the all-electron (AE) wave
function, a PAW correction needs to be added.  The correction will add the
cusp and all the wiggles necessary for the wave function to be orthogonal to
all the frozen core states.  This tutorial show how to use the
:class:`~gpaw.utilities.ps2ae.PS2AE` class for interpolating the PS wave
function to a fine grid where the PAW corrections can be represented:

.. autoclass:: gpaw.utilities.ps2ae.PS2AE

.. note:: Versions 20.10.0 and earlier

    The ``grid_spacing`` parameter was called ``h`` in older versions of GPAW.
    Using ``grid_spacing`` in the older versions will give a
    ``got an unexpected keyword`` error.

Here is the code for plotting some AE wave functions for a HLi dimer using a
PAW dataset for Li with a frozen 1s orbital
(:meth:`~gpaw.utilities.ps2ae.PS2AE.get_wave_function`):

.. literalinclude:: hli_wfs.py

.. figure:: hli-wfs.png

.. automethod:: gpaw.utilities.ps2ae.PS2AE.get_wave_function


.. _potential:

Electrostatic potential
-----------------------

The relevant formulas can be found here: :ref:`electrostatic potential`.

Here is how to extract the AE potential from a gpw-file using the
:meth:`~gpaw.utilities.ps2ae.PS2AE.get_electrostatic_potential` method:

.. literalinclude:: hli_pot.py

.. figure:: hli-pot.png

The figure also shows the average PS potentials at the atomic sites calculated
with the
:meth:`~gpaw.calculator.GPAW.get_atomic_electrostatic_potentials` method.

.. automethod:: gpaw.utilities.ps2ae.PS2AE.get_electrostatic_potential


Pseudo density
--------------

See:

.. automethod:: gpaw.utilities.ps2ae.PS2AE.get_pseudo_density
