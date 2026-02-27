.. module:: gpaw.response.gwqeh
.. _gwqeh tutorial:

===========================================================================
Band structure renormalization in van der Waals heterostructures: the
G\ :math:`\Delta`\ W method
===========================================================================

This tutorial demonstrates how to calculate quasiparticle energy
corrections for van der Waals (vdW) heterostructures using the
G\ :math:`\Delta`\ W-QEH method implemented in GPAW. The method computes
how the band structure of a 2D material is modified when it is placed
inside a vdW heterostructure, where the dielectric screening from
neighboring layers alters the screened Coulomb interaction.

For background on the GW approximation, see :ref:`gw tutorial` and
:ref:`gw_theory`. For the QEH model, see
:ref:`qeh`.


Physical motivation
===================

In a van der Waals heterostructure, the individual 2D layers interact
weakly through van der Waals forces. Because the interlayer coupling is
weak, the band structures of the individual layers are largely preserved.
However, the *dielectric environment* changes: a 2D material surrounded
by other polarizable layers experiences stronger screening of the Coulomb
interaction than the same material in isolation.

This has a direct consequence for quasiparticle energies. In the GW
approximation, quasiparticle energies depend on the screened Coulomb
interaction :math:`W`. When the screening is enhanced by neighboring
layers, the self-energy :math:`\Sigma = iGW` is modified, leading to
a **reduction of the band gap**. The effect can be substantial (hundreds
of meV) and is not captured by DFT, which does not account for
long-range dielectric screening from the environment.

Physically, this can be understood through the concept of *image
charges*: an electron (or hole) in one layer polarizes the neighboring
layers, creating an attractive image-charge potential that lowers the
quasiparticle energy of electrons and raises the energy of holes.
Because the image-charge interaction is attractive for both carriers,
the net effect is always a reduction of the band gap:

* **Valence band maximum (VBM)** shifts **up** (positive correction)
* **Conduction band minimum (CBM)** shifts **down** (negative correction)

The magnitude of the gap reduction scales inversely with the
polarizability of the material: less polarizable materials (larger
intrinsic band gaps) are more sensitive to changes in the dielectric
environment.


Theory
======

The G\ :math:`\Delta`\ W method computes the change in self-energy when
a monolayer is transferred from vacuum into a heterostructure. Rather
than performing a full GW calculation for the entire heterostructure
(which would be computationally prohibitive for incommensurate systems),
only the *change* in the screened interaction is needed.

The key quantity is the difference in screened Coulomb interaction:

.. math::

    \Delta W(\mathbf{q}, \omega) = W^{\mathrm{HS}}(\mathbf{q}, \omega)
    - W^{\mathrm{mono}}(\mathbf{q}, \omega)

where :math:`W^{\mathrm{HS}}` is the screened potential of the layer
embedded in the heterostructure and :math:`W^{\mathrm{mono}}` is the
screened potential of the isolated monolayer. Both are evaluated at the
position of the target layer using the QEH model, which constructs the
full dielectric response from precomputed *dielectric building blocks*
of the individual layers.

The QEH model works in two steps:

1. For each layer type, compute the density response function
   :math:`\chi^0(\mathbf{q}, \omega, z, z')` of the isolated layer
   and encode it as a "dielectric building block" (a ``.npz`` file
   containing the monopole and dipole components of the response).

2. Couple the building blocks through a macroscopic Dyson equation that
   accounts for the electrostatic interaction between layers, yielding
   the full screened interaction :math:`W^{\mathrm{HS}}` of the
   heterostructure.

The self-energy correction for band :math:`n\mathbf{k}` is then:

.. math::

    \Delta\Sigma_{n\mathbf{k}} = \frac{1}{\Omega}
    \sum_{\mathbf{q}}^{\mathrm{BZ}} \sum_{m}
    \frac{i}{2\pi} \int d\omega'\,
    |\rho^{n\mathbf{k}}_{m\mathbf{k-q}}|^2 \,
    \frac{\Delta W(\mathbf{q}, \omega')}
    {\varepsilon_{n\mathbf{k}} + \omega' - \varepsilon_{m\mathbf{k-q}}
    \pm i\eta}

where :math:`\rho^{n\mathbf{k}}_{m\mathbf{k-q}}` are the pair density
matrix elements from the monolayer calculation. This has the same
structure as the standard GW self-energy (see :ref:`gw_theory`), but
with :math:`W` replaced by :math:`\Delta W`.

Finally, the quasiparticle correction is obtained as:

.. math::

    \Delta E^{\mathrm{QP}}_{n\mathbf{k}} = Z_{n\mathbf{k}} \cdot
    \mathrm{Re}\, \Delta\Sigma_{n\mathbf{k}}

where :math:`Z_{n\mathbf{k}}` is the quasiparticle renormalization
factor. If a GW calculation for the monolayer is available,
:math:`Z` is taken from that calculation. Otherwise, a default value
of :math:`Z = 0.7` is used, which is a reasonable approximation for
most 2D semiconductors [#Rasmussen2021]_.

The G\ :math:`\Delta`\ W correction is **k-independent** to a good
approximation, because :math:`\Delta W(\mathbf{q}, \omega)` is a smooth
function of :math:`\mathbf{q}` with weak spatial structure. This means
it is usually sufficient to evaluate the correction at a single k-point
and apply it as a rigid shift to the monolayer band structure.


References
----------

.. [#Winther2017] K. T. Winther and K. S. Thygesen,
   "Band structure engineering in van der Waals heterostructures via
   dielectric screening: the G\ :math:`\Delta`\ W method",
   *2D Materials* **4**, 025059 (2017).
   https://doi.org/10.1088/2053-1583/aa6531

.. [#Thygesen2017] K. S. Thygesen,
   "Calculating excitons, plasmons, and quasiparticles in 2D materials
   and van der Waals heterostructures",
   *2D Materials* **4**, 022004 (2017).
   https://doi.org/10.1088/2053-1583/aa6432

.. [#Rasmussen2021] A. Rasmussen *et al.*,
   "Efficient many-body calculations for two-dimensional materials using
   exact limits for the screened potential: Band gaps of MoS\ :sub:`2`,
   h-BN, and phosphorene",
   *Phys. Rev. B* **94**, 155406 (2016).

.. [#Leon2025] D. A. Leon *et al.*,
   "Self-consistent layer-projected scissors operator for band structures
   of two-dimensional van der Waals materials with large unit cells",
   *Phys. Rev. B* **112**, 115128 (2025).
   https://doi.org/10.1103/2y5s-qcw9


Prerequisites
=============

The G\ :math:`\Delta`\ W-QEH calculation requires:

1. A DFT groundstate calculation for the monolayer of interest,
   with full diagonalization to obtain all required bands.
2. A dielectric building block file (``chi.npz``) for each layer
   type in the heterostructure.
3. The ``qeh`` Python package (install via ``pip install qeh``).

The workflow consists of four steps:


Step 1: Groundstate calculation
===============================

First, perform a DFT groundstate calculation for a monolayer of
MoS\ :sub:`2`. We use a plane-wave basis with LDA exchange-correlation:

.. literalinclude:: MoS2_groundstate.py

Key points:

* The structure is created using ASE's ``mx2`` builder with the
  experimental lattice constant *a* = 3.184 Ang.
* We use ``diagonalize_full_hamiltonian()`` to obtain unoccupied states
  needed for the GW self-energy.
* The file is saved with ``mode='all'`` to include wavefunctions.


Step 2: Create dielectric building block
=========================================

Next, compute the dielectric building block for the QEH model. This
involves calculating the (q, omega)-dependent density response of the
isolated monolayer:

.. literalinclude:: MoS2_buildingblock.py

The building block file (``MoS2-chi.npz``) contains the monopole and
dipole components of the layer's polarizability as a function of
in-plane momentum :math:`q` and frequency :math:`\omega`.

.. note::

    The building block calculation is the most computationally demanding
    step. The parameters ``ecut`` and ``kpts`` in the groundstate, as
    well as the frequency grid parameters, should be converged for
    production calculations. See the convergence section below.


Step 3: G\ :math:`\Delta`\ W-QEH calculation
=============================================

Now we set up the G\ :math:`\Delta`\ W calculation. In this example,
we compute the quasiparticle correction for one MoS\ :sub:`2` layer
in a bilayer MoS\ :sub:`2`/MoS\ :sub:`2` homostructure:

.. literalinclude:: MoS2_gwqeh.py

The key parameters are:

* ``calc``: Path to the groundstate ``.gpw`` file of the monolayer.
* ``bands``: Range of band indices to compute corrections for.
  Should include both valence and conduction bands around the gap.
* ``structure``: List of building block files defining each layer in
  the heterostructure, from bottom to top.
* ``d``: List of interlayer distances (in Angstrom) between
  consecutive layers. Length is ``len(structure) - 1``.
* ``layer``: Index (0-based) of the layer to compute corrections for.
* ``kpts``: k-point indices. Typically ``[0]`` (Gamma) is sufficient
  since the correction is approximately k-independent.

.. note::

    The quasiparticle correction :math:`\Delta E^{\mathrm{QP}}` is
    approximately k-independent because :math:`\Delta W` has weak
    spatial structure. This means you can evaluate it at a single
    k-point (typically Gamma) and apply it as a rigid shift to the
    full band structure obtained from a monolayer GW calculation.


Step 4: Analyze and plot results
================================

The quasiparticle corrections can be extracted and visualized:

.. literalinclude:: MoS2_analyze.py

This script loads the results and produces a bar chart showing the
QP correction for each band. Valence bands should have positive
corrections (shifting up) and conduction bands should have negative
corrections (shifting down), consistent with the band gap reduction
from environmental screening.

.. image:: MoS2_qp_corrections.png

The corrections are typically on the order of 50--200 meV per band
for TMD bilayers, corresponding to a total band gap reduction of
0.1--0.4 eV.


Convergence considerations
==========================

For production calculations, the results must be converged with respect
to:

1. **k-point sampling**: The dielectric building block should be
   converged with respect to k-points (``kpts`` in the groundstate
   calculation). Typical converged values for TMDs are 12x12x1 to
   18x18x1.

2. **Response energy cutoff**: The ``ecut`` parameter in the building
   block calculation controls the number of plane waves used to
   represent the response function. Values of 50--150 eV are typical.

3. **q-point grid**: The QEH model uses a q-point grid determined by
   the ``q_max`` parameter. It should be large enough to capture the
   q-dependence of the screened interaction.

4. **Frequency grid**: The parameters ``domega0`` and ``omega2`` control
   the non-linear frequency grid for the self-energy integration.
   ``domega0`` sets the minimum spacing near :math:`\omega = 0`, and
   ``omega2`` controls where the spacing doubles.

5. **Number of bands**: The bands included in the self-energy sum
   (controlled by ``nbands`` in the groundstate) should be converged.


Example: Combining with monolayer GW
=====================================

For quantitative results, the G\ :math:`\Delta`\ W-QEH correction
should be combined with a full GW calculation for the isolated monolayer.
The ``gwfile`` parameter provides the monolayer GW results, from which
the renormalization factor :math:`Z` is extracted:

.. literalinclude:: MoS2_gwqeh_with_gw.py

This uses the quasiparticle renormalization factor from the GW
calculation rather than the default estimate of :math:`Z = 0.7`. The
full quasiparticle energies are then:

.. math::

    E^{\mathrm{QP, HS}}_{n\mathbf{k}} = E^{\mathrm{QP, mono}}_{n\mathbf{k}}
    + \Delta E^{\mathrm{QP}}_{n\mathbf{k}}


Parallelization
===============

The G\ :math:`\Delta`\ W calculation can be parallelized over k-points
and bands:

.. code-block:: bash

    mpirun -np 4 gpaw python MoS2_gwqeh.py

The building block calculation (Step 2) is typically the most
computationally demanding part and benefits most from parallelization.
