.. _scissors operator:

===============================
Scissors operator for LCAO mode
===============================

.. warning:: **Work in progress**

.. module:: gpaw.lcao.scissors
.. autofunction:: non_self_consistent_scissors_shift

In :ref:`lcao` we solve the following generalized eigenvalue problem:

.. math::

 \sum_\nu (H + \Delta H)_{\mu\nu} C_{\nu n}
 = \sum_{\nu} S_{\mu\nu} C_{\nu n} \epsilon_n,

where `\Delta H` is a scissors operator.

Space is divided into regions `\Omega_i` and for each region we define desired
shifts of the occupied and unoccupied bands: `\Delta^i_{\text{occ}}` and
`\Delta^i_{\text{unocc}}`.

For each region, we diagonalize the density-matrix

.. math::

  \rho_{\mu\nu} =
  \sum_n C_{\mu n} f_n C_{\nu n}^*

in the orbitals belonging to `\Omega_i`:

.. math::

   \sum_{\nu\in i}(S^{1/2}\rho S^{1/2})_{\mu\nu} V^i_{\nu\alpha} =
   \lambda^i_\alpha V^i_{\mu\alpha}.

Here, the eigenvalues `\lambda^i_\alpha` will be close to either zero or one and
the scissors operator is now given as:

.. math::

    \Delta H_{\mu\nu} = \sum_i \sum_{\mu'\in i,\nu'\in i} \sum_\alpha
    S^{1/2}_{\mu\mu'} V^i_{\mu'\alpha}
    (\lambda^i_\alpha \Delta^i_{\text{occ}} +
     (1 - \lambda^i_\alpha) \Delta^i_{\text{unocc}})
    V^{i*}_{\nu'\alpha} S^{1/2*}_{\nu'\nu}.


.. _scissors band structure:

WS2 layer on top of MoS2 layer
==============================

Band structures for:

* no shifts
  (``shifts=[]``)
* MoS2 gap opened up by 1.0 eV
  (``shifts=[(-0.5, 0.5, 3)]``)
* MoS2 shifted up by 0.5 eV and WS2 down by 0.5 eV
  (``shifts=[(0.5, 0.5, 3), (-0.5, -0.5, 3)]``)


.. figure:: mos2ws2.png

.. literalinclude:: mos2ws2.py

.. literalinclude:: plot_bs.py

.. tip::

   You can plot the JSON band-structure files with the command:
   ``ase band-strucuture <name>.json``.
