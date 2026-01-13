.. _qeh tutorial:
.. module:: gpaw.response.qeh

=====================================================
The Quantum Electrostatic Heterostructure (QEH) model
=====================================================

Brief overview
==============

In this tutorial we show how to calculate the linear response of a general van
der Waals Heterostructure (vdWH) by means of the quantum electrostatic
heterostructure (QEH) model. This method allows to overcome the computational
limitation of the standard ab-initio approaches by combining quantum accuracy
at the monolayer level and macroscopic electrostatic coupling between the
layers. More specifically, the first step consists in calculating the response
function of each of the layers composing the vdWHs in the isolated condition and
encoding it into a so called dielectric building block. Then, in the next step
the dielectric building blocks are coupled together through a macroscopic Dyson
equation. The validity of such an approach strongly relies on the absence of
hybridization among the layers, condition which is usually satisfied by vdWHs.

A thorough description of the QEH model can be found in [#qeh_theory]_
and [#qeh_theory2]_:

.. [#qeh_theory] K. Andersen, S. Latini and K.S. Thygesen
    :doi:`Dielectric genome of van der Waals heterostructures
    <https://doi.org/10.1021/acs.nanolett.5b01251>`
    *Nano Letters* **15** (7), 4616-4621 (2015)

.. [#qeh_theory2]
    M. N. Gjerding, L. S. R. Cavalcante, Andrey Chaves, and K. S. Thygesen
    :doi:`Efficient Ab Initio Modeling of Dielectric Screening in
    2D van der Waals Materials: Including Phonons, Substrates, and Doping
    <https://doi.org/10.1021/acs.jpcc.0c01635>`
    J. Phys. Chem. C 2020, 124, 21, 11609–11616


.. seealso::

    * `Documentation <https://qeh.readthedocs.io/en/latest/>`_
      for the ``qeh`` Python package.
    * `qeh on Gitlab <https://gitlab.com/camd/qeh/>`_.
    * Examples from the
      `CMR database <https://cmr.fysik.dtu.dk/vdwh/vdwh.html>`_.


Constructing a dielectric building block
========================================

First, we need a ground state calculation for each of the layers in the vdWH.
We will consider a MoS2/WSe2 bilayer. In the following script
we show how to get the ground state gpw-file for the MoS2 layer:

.. literalinclude:: gs_MoS2.py

The gpw-file for WSe2 can be obtained in a similar way.
The gpw-files stores all the necessary eigenvalues and eigenfunctions for the
response calculation.

Next, we need the `z`-dependence of the full response function 
`\chi(q,\omega, G_z, G'_z)`. Essentially, a Dyson equation for the isolated layer
is solved to obtain the full response function `\chi(q,\omega,G,G')`, and then the
desired `G_z`-elements are chosen from its basis. This is all handled by the ChiHandler
object, and saved to the file \*-chi.npz. Subsequently, the method interpolate_chi_to_bb
will interpolate the full response function onto a dense `q`-grid, and calculate
the monopole and dipole component of the response function and the corresponding
induced densities. These quantities are then condensed into the dielectric
building block \*-bb.npz. Here is how to get the MoS2 and building block:

.. literalinclude:: bb_MoS2.py

The same goes for the WSe2 building block.
Once the building blocks have been created, they need to be interpolated to the
same kpoint and frequency grid. This is done as follows:

.. literalinclude:: interpolate_bb.py

Specifically, this interpolates both building block to lowest `q`-grid of the two.

Finally the building blocks are ready to be combined electrostatically.


Interlayer excitons in MoS2/WSe2
================================

As shown in [#interlayer]_ the MoS2_WSe2 can host excitonic excitations where
the electron and the hole are spatially separated, with the electron sitting
in the MoS2 layer and the hole in the WSe2 one.

Because of the finite separation distance we expect that the electron-hole
screened interaction between the two particles will not diverge when the
in-plane separation goes to 0. To illustrate this we show how to calculate
the screened electron-hole interaction using the QEH model and the building
blocks created above:


.. literalinclude:: interlayer.py
    :end-before: get_exciton_binding_energies

Here is the generated plot:

.. image:: W_r.png

As expected the interaction does not diverge.

If one is also interested in the interlayer exciton binding energy, it can be
readily calculated by appending the following lines in the script above:

.. literalinclude:: interlayer.py
    :start-after: show

We find an interlayer exciton binding energy of `\sim 0.3` eV.


.. [#interlayer] S. Latini, K.T. Winther, T. Olsen and K.S. Thygesen
   Interlayer Excitons and Band Alignment in MoS2/hBN/WSe2
   van der Waals Heterostructures
   *Nano Letters* Just accepted (2017)
