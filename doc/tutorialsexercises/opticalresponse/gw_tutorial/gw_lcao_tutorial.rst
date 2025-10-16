==================================================================================
Quasi-particle spectrum in the GW approximation from LCAO wave functions: tutorial
==================================================================================

Groundstate calculation
-----------------------

First, we need to do a regular LCAO groundstate calculation, and save all of the
wave functions.

.. literalinclude:: C_lcao_groundstate.py

G0W0 calculation using LCAO basis functions
-------------------------------------------

We can now set up the G0W0 calculation rather normally. However,
there are a few major differences. `ecut_extrapolation` is disabled,
because the convergence wrt. G-bands and chi0 bands is unknown with LCAO-basis set,
whereas in planewave mode, the number of bands can be just directly chosen from
the plane wave cut off.


