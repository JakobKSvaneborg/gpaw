.. module:: gpaw.response.bse
.. _bseplus tutorial:

==================
The BSE+ equations
==================

In this tutorial, we calculate the BSE+ [#Sondersted]_  q-resolved electron energy loss spectrum (q-EELS) of TiO2 and MoS2, as well as the refractive index for TiO2.

Absorption spectrum of TiO2
===========================

BSE+ needs two kinds of ground state ``gpw``-files as input. One with fewer k-points to calculate the BSE, and one with more k-points and bands for the RPA calculations.

The files created will be called ``fixed_density_calc_TiO2_bse.gpw``
which will contain fewer k-points for the BSE calculation itself. 
Note, that BSE calculation will require many bands to calculate the screened interaction W, which
demands a significant number of unoccupied bands.

The second file, ``fixed_density_calc_TiO2_rpa.gpw``, will contain even more bands for the RPA calculation and a higher k-point density to ensure better convergence.

This calculation will run 20mins with 80 cores.

.. literalinclude:: gs_TiO2.py

Now we are ready to perform the actual BSE+ calculation.
We instantiate the ``BSEPlus`` object, which requires these two ``gpw``-files: one for the BSE calculation, and one for the RPA calculation. Parameters starting with ``bse_`` are related to the BSE calculation, and should be se as in a standard BSE calculation. We use 60 bands to calculate the screening (W) for BSE, and 130 bands for the RPA calculation. We shift the bandgap to match the direct band gap of TiO2, aligning the lowest peak observed in the refractive index calculated with BSE with the lowest peak in the refractive index observed in the experimental data. 

This calculation will run 2 hours with 80 cores.

.. literalinclude:: BSEPlus_TiO2.py

Now we can plot the results together with the experimental data which can be downloaded 
from here :download:`tio2_n_rutile_inplane.csv` [#Jellison]_, :download:`eels_tio2_rutile.csv` [#Launay]_.

.. literalinclude:: plot_TiO2.py

.. image:: n_TiO2.png

.. image:: eels_TiO2.png


Absorption spectrum of MoS2
===========================

BSE+ can also be used to calculate q-EELS spectra of 2D materials. Here, we calculate the EELS for a slab of MoS2 at `q=0.074 Ao^{-1}`.

As before, the files ``fixed_density_calc_MoS2_bse.gpw`` and ``fixed_density_calc_MoS2_rpa.gpw`` will be created with the former containing fewer k-points for the BSE calculation. The k-point densities are chosen such that the BSE k-point grid is contained in the RPA k-point grid to allow for finite q calculations.

This calculation will run 35mins with 80 cores and requires the structure file for MoS2 :download:`MoS2.json`.

.. literalinclude:: gs_MoS2.py

We are now ready to do the BSE+ calculation. This time, we use a truncated Coulomb kernel and include spin-orbit coupling in the BSE calculation.

This calculation will run 2 hours with 80 cores.

.. literalinclude:: BSEPlus_MoS2.py

We can then plot the resulting EELS spectrum. We create two plots, one showing a broad frequency range and anothe zoomed in on the x-axis to visualize the spin-orbit-split exciton below the band edge. The results are shown together with the experimental data which can be downloaded from here :download:`MoS2_q_0p060Ainv.csv` [#Koster]_.

.. literalinclude:: plot_MoS2.py

.. image:: eels_MoS2_low_frequencies.png

.. image:: eels_MoS2.png



.. [#Sondersted] Søndersted et al.
                 *Phys. Rev. Lett.* **133**, 026403 (2024)
.. [#Jellison] Jellison et al.
               *J. Appl. Phys.* **93**, 9537 (2003)
.. [#Launay] Launay et al.
            *Phys. Rev. B* **69**, 035101 (2004)
.. [#Koster] Köster et al.
            *Micron* **106**, 103303 (2022)

