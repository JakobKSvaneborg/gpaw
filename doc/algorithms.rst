.. _features and algorithms:

=======================
Features and algorithms
=======================

Quick links to all features:

.. list-table::

    * - :ref:`Plane-waves <manual_mode>`
      - :ref:`Finite-difference <manual_stencils>`
      - :ref:`LCAO <lcao>`
    * - :ref:`XC-functionals <xc>`
      - :ref:`DFT+U <hubbardu>`
      - :ref:`GLLB-SC <band_gap>`
    * - :ref:`DOS <dos>`
      - :ref:`STM <stm tutorial>`
      - :ref:`Wannier functions <wannier>`
    * - :ref:`delta-SCF <dscf>`
      - :ref:`XAS <xas>`
      - :ref:`Jellium <jellium>`
    * - :ref:`TDDFT <timepropagation>`
      - :ref:`LRTDDFT (molecules) <lrtddft>`
      - :ref:`LRTDDFT (extended systems) <df_theory>`
    * - :ref:`RPA-correlation <rpa>`
      - :ref:`GW <gw_theory>`
      - :ref:`BSE <bse theory>`
    * - :ref:`Parallelization <parallel_runs>`
      - :ref:`Continuum Solvent Model <continuum_solvent_model>`
      - :ref:`point groups`

This Page gives a quick overview of the algorithms used.  We have
written some :ref:`papers <faq>` about the implementation,
where *all* the details can be found.


**Introduction**

Using the projector-augmented wave (PAW)
method [Blo94]_, [Blo03]_  allows us to get rid of the core
electrons and work with soft pseudo valence wave functions.  The
pseudo wave functions don't need to be normalized - this is important
for the efficiency of calculations involving 2. row elements (such as
oxygen) and transition metals.  A further advantage of the PAW method
is that it is an all-electron method (frozen core approximation) and
there is a one to one transformation between the pseudo and
all-electron quantities.


**Description of the wave functions**

Pseudo wave functions can be described in three ways:

Plane-waves (PW):
    Expansion in plane-waves.

Linear combination of atomic orbitals (LCAO):
    Expansion in atom-centered basis functions.

Finite-difference (FD):
    Uniform real-space grids.


**ASE interface**

The code has been designed to work together with the atomic
simulation environment (`ASE <https://ase-lib.org>`_). ASE provides:

 * Structure optimization.
 * Molecular dynamics.
 * Nudged elastic band calculations.
 * Maximally localized Wannier functions.
 * Scanning tunneling microscopy images.


**Open Software**

GPAW is released under the :xkcd:`GNU Public License <225>`
version 3 or any later version.  See the file :git:`LICENSE` which
accompanies the downloaded files, or see the license at GNU's web
server at https://www.gnu.org/licenses/.  Everybody is invited to
participate in using and :ref:`developing the code <devel>`.


.. figure:: carlsberg.png
    :width: 12cm

    September 2003 - August 2005: Sponsored by The `Carlsberg Foundation`_
    (artwork by P. Erhart)

.. _Carlsberg Foundation: https://www.carlsbergfondet.dk


.. [Blo94] P. E. Blöchl,
   Phys. Rev. B 50, 17953 (1994)
.. [Blo03] P. E. Blöchl, C. J. Först and J. Schimpl,
   Bull. Mater. Sci, 26, 33 (2003)
