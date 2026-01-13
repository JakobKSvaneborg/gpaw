.. module:: gpaw.response.gwqeh
.. _gwqeh tutorial:

===========================================================================
Quasiparticle corrections for van der Waals heterostructures: GWQEH method
===========================================================================

This tutorial demonstrates how to calculate quasiparticle energies for
van der Waals heterostructures using the GWQEH method. The GWQEH method
combines GW calculations for individual layers with the Quantum
Electrostatic Heterostructure (QEH) model to account for the modified
dielectric screening in the heterostructure environment.

For background on the GW approximation, see :ref:`gw tutorial` and
:ref:`gw_theory`.

The GWQEH method is particularly useful for:

* Calculating band alignments in van der Waals heterostructures
* Determining quasiparticle corrections due to environmental screening
* Studying interlayer effects on electronic structure

Theory
======

In a van der Waals heterostructure, the dielectric environment of each
layer is modified by the presence of neighboring layers. This affects
the screened Coulomb interaction and consequently the quasiparticle
energies.

The GWQEH method calculates the quasiparticle correction as:

.. math::

    \Delta E^{QP} = Z \cdot \Delta\Sigma

where :math:`\Delta\Sigma` is the change in self-energy due to the
modified screening, and :math:`Z` is the quasiparticle renormalization
factor.

The change in screened interaction :math:`\Delta W` is obtained from
the QEH model, which combines dielectric building blocks from individual
layer calculations.

More information can be found in:

    \K. S. Thygesen

    `Calculating excitons, plasmons, and quasiparticles in 2D materials
    and van der Waals heterostructures`__

    2D Materials, Vol. **4**, 022004 (2017)

    __ https://doi.org/10.1088/2053-1583/aa6432

Prerequisites
=============

The GWQEH calculation requires:

1. A groundstate calculation for the layer of interest
2. A dielectric building block file (chi.npz) for each layer in the
   heterostructure
3. The ``qeh`` Python package (install via ``pip install qeh``)

Step 1: Groundstate calculation
===============================

First, perform a DFT groundstate calculation for a monolayer of MoS2.
We use a plane-wave basis with sufficient k-point sampling:

.. literalinclude:: MoS2_groundstate.py

This creates a gpw file with all the wavefunctions needed for subsequent
calculations.

Step 2: Create dielectric building block
========================================

Next, create a dielectric building block file that will be used by the
QEH model. This requires calculating the dielectric function of the
isolated layer:

.. literalinclude:: MoS2_buildingblock.py

The building block file (``MoS2-chi.npz``) contains the polarizability
of the layer as a function of in-plane momentum and frequency.

.. note::

    The building block calculation can take some time depending on the
    k-point sampling and energy cutoff. For production calculations,
    convergence with respect to these parameters should be checked.

Step 3: GWQEH calculation
=========================

Now we can set up and run the GWQEH calculation. In this example, we
calculate the quasiparticle correction for a MoS2 layer in a bilayer
MoS2/MoS2 heterostructure:

.. literalinclude:: MoS2_gwqeh.py

The key parameters are:

* ``calc``: Path to the groundstate gpw file
* ``bands``: Range of bands to calculate (valence and conduction bands)
* ``structure``: List of building block files defining the heterostructure
* ``d``: Interlayer distances in Angstrom
* ``layer``: Index of the layer to calculate corrections for
* ``kpts``: k-points for the calculation (typically [0] is sufficient)

.. note::

    The quasiparticle correction is often similar for different k-points,
    so calculating at the Gamma point (kpts=[0]) is usually sufficient.
    The correction can then be applied to band structures from standard
    GW calculations.

Step 4: Analyze results
=======================

The quasiparticle corrections can be extracted and analyzed:

.. literalinclude:: MoS2_analyze.py

Expected output::

    Band index: 8, QP correction: -0.15 eV
    Band index: 9, QP correction: -0.12 eV
    Band index: 10, QP correction: 0.08 eV
    Band index: 11, QP correction: 0.10 eV

.. note::

    The above values are examples. Actual values will depend on the
    specific heterostructure geometry and convergence parameters.

Convergence considerations
==========================

For production calculations, the following convergence checks are
recommended:

1. **k-point sampling**: The dielectric building block should be
   converged with respect to k-points.

2. **Energy cutoff**: The ``ecut`` parameter in the building block
   calculation affects the accuracy.

3. **q-point grid**: The QEH model uses a q-point grid that should be
   dense enough to capture the q-dependence of the screened interaction.

4. **Frequency grid**: The parameters ``domega0`` and ``omega2`` control
   the frequency grid for the self-energy calculation.

Example: MoS2/WSe2 heterostructure
==================================

Here is a more complete example for a MoS2/WSe2 bilayer heterostructure:

.. literalinclude:: MoS2_WSe2_heterostructure.py

This demonstrates how to combine building blocks from different materials
to study band alignment in a realistic heterostructure.

Including GW results
====================

For more accurate results, you can combine the GWQEH correction with
a full GW calculation for the isolated layer. The ``gwfile`` parameter
allows you to provide the GW results file:

.. literalinclude:: MoS2_gwqeh_with_gw.py

This uses the quasiparticle renormalization factor from the GW calculation
instead of the default estimate of Z=0.7.

Parallelization
===============

The GWQEH calculation can be parallelized over k-points and bands. For
large systems, running on multiple cores can significantly reduce
computation time:

.. code-block:: bash

    mpirun -np 4 gpaw python MoS2_gwqeh.py

.. note::

    The building block calculation (Step 2) is typically the most
    computationally demanding part and benefits most from parallelization.
