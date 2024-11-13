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
shifts of the occupied and unoccupied bands: `\Delta_{i,\text{occ}}` and
`\Delta_{i,\text{unocc}}`.  The scissors operator is given as:

.. math::

    \Delta H = \sum_i(\Delta H^{i,\text{occ}}+\Delta H^{i,\text{unocc}}),

where the `\mu,\nu\in\Omega_i` blocks of
`\Delta H^{i,\text{occ}}_{\mu\nu}` and
`\Delta H^{i,\text{unocc}}_{\mu\nu}` are given by:

.. math::

    \Delta H_{\mu\nu}^{i,\text{occ}} =
        \Delta_{i,\text{occ}}
        \sum_n^{\text{occ}}
        \sum_{\mu'\nu'}
        S_{\mu\nu'}^*
        C_{\nu'n}
        C_{\mu'n}^*
        S_{\mu'\nu},

.. math::

    \Delta H_{\mu\nu}^{i,\text{unocc}} =
        \Delta_{i,\text{unocc}}
        \sum_n^{\text{unocc}}
        \sum_{\mu'\nu'}
        S_{\mu\nu'}^*
        C_{\nu'n}
        C_{\mu'n}^*
        S_{\mu'\nu}.


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
