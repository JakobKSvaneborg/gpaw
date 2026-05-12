.. _bandstructures:

=========================================
Calculation of electronic band structures
=========================================

In this tutorial we calculate the electronic band structure of Si along
high symmetry directions in the Brillouin zone.

First, a standard ground state calculations is performed and the results
are saved to a *.gpw* file. As we are dealing with small bulk system,
plane wave mode is the most appropriate here.

.. literalinclude:: bandstructure.py
    :start-after: P1
    :end-before: P2

Next, we calculate eigenvalues along a high symmetry path in the Brillouin
zone ``kpts={'path': 'GXWKL', 'npoints': 60}``.  See
:data:`ase.dft.kpoints.special_points` for the definition of the special
points for an FCC lattice.

For the band structure calculation, the density is fixed to the previously
calculated ground state density, and as we want to
calculate all k-points, symmetry is not used (``symmetry='off'``).

.. literalinclude:: bandstructure.py
    :start-after: P2
    :end-before: P3

Finally, the bandstructure can be plotted (using ASE's band-structure tool
:class:`ase.spectrum.band_structure.BandStructure`):

.. literalinclude:: bandstructure.py
    :start-after: P3

.. figure:: bandstructure.png

The full script: :download:`bandstructure.py`.


Effect of spin-orbit coupling
=============================

Here is a zoom in on the VBM to see the effect of including
:ref:`spinorbit`:

.. figure:: si-soc-bs.png

.. literalinclude:: soc.py
    :start-after: web-page


.. _hse06 on lda:

Non self-consistent HSE06
=========================

.. literalinclude:: hse06.py
.. figure:: hse06.png
.. literalinclude:: plot_hse06.py
.. autoclass:: gpaw.hybrids.NonSelfConsistentHybridXCCalculator
   :members:
.. autofunction:: gpaw.new.pw.hybrids.truncated_coulomb
