.. _gw_lcao_tutorial:

==================================================================================
Quasi-particle spectrum in the GW approximation from LCAO wave functions: tutorial
==================================================================================

For a brief introduction to the GW theory and the details of its
implementation in GPAW, see :ref:`gw_theory`.

More information can be found here:

    \F. Hüser, T. Olsen, and K. S. Thygesen

    `Quasiparticle GW calculations for solids, molecules, and
    two-dimensional materials`__

    Physical Review B, Vol. **87**, 235132 (2013)

    __ https://prb.aps.org/abstract/PRB/v87/i23/e235132
    
Additionally, before getting started using LCAO wave functions in the GW approximation,
it would be beneficial to become familiar with G0W0 using GPAW, see :ref:`gw_tutorial`.

Quasi-particle spectrum of bulk diamond
=======================================


Groundstate LCAO calculation
----------------------------

First, we need to do a regular LCAO groundstate calculation, and save all of the
wave functions. The basis-set is chosen using the ``basis`` keyword, and  
``nbands`` is set to ``nao``, reflecting the maximum nbands value that can be used in LCAO,
which is the same number of bands as there are atomic orbitals. 

.. literalinclude:: C_lcao_groundstate.py

G0W0 calculation using LCAO basis functions
-------------------------------------------

We can now set up the G0W0 calculation similar to the PW case, passing the LCAO groundstate 
calculation ``C_lcao_groundstate.gpw`` to the G0W0 calculator, which is then internally converted
to PW. 
However, there are a few major differences. ``ecut_extrapolation`` is disabled,
because the convergence wrt. G-bands and chi0 bands is unknown with LCAO-basis set,
whereas in PW mode, the number of bands can be just directly chosen from
the plane wave cut off.

.. literalinclude:: C_lcao_gw.py

The results are stored in ``C-g0w0-lcao_results.pckl``.  

GW self-energy for Diamond (C)
------------------------------

We plot the results stored in ``C-g0w0-lcao_results.pckl`` as the imaginary and real part of the 
GW self-energy :math:`\Sigma(\omega)` depending on the frequency using the script 
:download:`plot_C_lcao_gw.py`. We compare the LCAO results to PW ones computed 
using :download:`C_pw_groundstate.py` and :download:`C_pw_gw.py`. 

.. figure:: C_Im.png
   :align: center
   :width: 50%

.. figure:: C_Re.png
   :align: center
   :width: 50%



