.. module:: gpaw.response.bse
.. _bseplus tutorial:

==================
The BSE+ equations
==================

In this tutorial, we calculate the BSE+ [#Sondersted]_
the EELS spectrum of TiO2 and MoS2, and also
the refractive index for TiO2.

Absorption spectrum of TiO2
===========================

BSE+ needs two kinds of ground state ``gpw``-files as an input.
One with fewer k-points to calculate the BSE,
and one with more k-points and bands for the RPA calculations.

The files created will be called ``fixed_density_calc_TiO2_bse.gpw``
which will contain less k-points for the BSE calculation itself. 
Note, that BSE calculation will require to calculate W, so this
also will need a signifficant amount of unoccupied bands.
The second file created, from the same system ``fixed_density_calc_TiO2_rpa.gpw`` will contain even more bands for the RPA calculation, but also will have higher k-point density for better convergence.

This calculation will run 30mins with 24 cores.

.. literalinclude:: gs_TiO2.py

Now we are ready to perform the actual BSE+ calculation.
We instantiate the ``BSEPlus`` object, and it requires these two
``gpw``-files, one for the BSE calculation, and one for the RPA calculation. The parameters starting with ``bse_`` are relayed to the BSE-calculation, so we choose these as we would in a normal BSE-calculation.
We use 60 bands to calculate screening (W) for BSE, and 130 bands for the RPA calculation. We shift the bandgap to match the direct band gap of TiO2.

This calculation will run 2 hours with 80 cores.

.. literalinclude:: BSEPlus_TiO2.py

Now we can plot the results together with the experimental data which can be downloaded 
from here :download:`tio2_n_rutile_inplane.csv`, :download:`eels_tio2_rutile.csv`.

.. literalinclude:: plot_TiO2.py

.. image:: n_TiO2.png

.. image:: eels_TiO2.png


Absorption spectrum of MoS2
===========================

.. literalinclude:: gs_MoS2.py


.. [#Sondersted] Søndersted et al.
                 *Phys. Rev. Lett.* **133**, 026403 (2024)

