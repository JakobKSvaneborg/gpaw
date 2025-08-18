.. module:: gpaw.berryphase
.. _berry tutorial:

==========================
 Berry phase calculations
==========================

In this tutorial we provide a set of examples of how to calculate various
physical properties derived from the `k`-space Berry phases.


Spontaneous polarization of tetragonal BaTiO3
=============================================

As a first example we calculate the spontaneous polarization of the
ferroelectric BaTiO3 in the tetragonal phase. The initial step is to generate
a ground state calculation with wavefunctions. This can be don e with the
script :download:`gs_BaTiO3.py`. We can then run the script

.. literalinclude:: polarization_BaTiO3.py

which calculates the polarization. It will take a few minutes on a single CPU,
but can also be parallelized. The calculation adds the contribution from the
electrons and the nucleii, which implies that the result is independent of the
positions of the atoms relative to the unit cell. The results should be 0.27
`C/m^2` for LDA and 0.45 `C/m^2` for PBE , which agrees with the values from
literature [#Zhang]_.  Technically this approach is incorrect since an
adiabatic path between a non-polar reference structure (in this case the cubic
phase of BaTiO3) and the polar phase (the tetragonal phase of BaTiO3) is
needed in order to properly compute the spontanous polarization [#King-Smith]_.
However for BaTiO3 the spontaneous polarization happens to be smaller
than the so called polarization quantum, and therefore the approach presented
here yields the correct result.


Born effective charges of tetragonal BaTiO3
===========================================

The Berry phase module can also be used to calculate Born effective charges.
They are defined by

.. math:: Z^a_{ij}=V\frac{\partial P_i}{\partial u_j^a}

where `u_j^a` is the position of atom  `a` in direction `j` relative to the
equilibrium position and `P_i` is the polarization in direction `i`. Like the
case of phonons all atoms are moved in all possible directions and the induced
polarization is calculated for each move. For the BaTiO3 structure calculated
above the calculation is performed with the script

.. literalinclude:: born_BaTiO3.py

The results are written to the ``born_charges.json`` file and the Born effective charges
can be viewed with the script

.. literalinclude:: get_borncharges.py
    :start-after: literalinclude import-start
    :end-before: literalinclude import-end

Due to symmetry all the tensors are diagonal. Note, however, the large
differences between the components for each of the O atoms. The Born effective
charges tell us how the atoms are affected by an external electric field.

.. csv-table::
  :align: center
  :header: Ba, , , ,Ti, , , ,O, , , ,O, , , ,O, , ,
  :file: born_charges_BaTiO3.csv

Topological properties of stanene from parallel transport
=========================================================

We will now demonstrate how the `k`-space Berry phases can be applied
to extract topological properties of solids. For an isolated band we can
calculate the Berry phase

.. math:: \gamma_1(k_2, k_3)=i\int_0^1 dk_1 \langle u(\mathbf{k})|\partial_{k_1}|u(\mathbf{k})\rangle

where `k_i` is the component of crystal momentum corresponding to
`\mathbf{b}_i` in reduced coordinates. The Berry phase is only gauge invariant
modulo `2\pi` and since
`\gamma_1(k_2, k_3)=\gamma_1(k_2+1, k_3)=\gamma_1(k_2,k_3+1)` modulo `2\pi`,
we can count how many multiples of the `2\pi` that
`\gamma_1` changes while `k_2` of `k_3` is adiabatically cycled through the
Brillouin zone. This is the Chern number, which gives rise to a topological
`\mathbb{Z}` classification of all two-dimensional insulators. For multiple
valence bands the situation is slightly more complicated and one has to
introduce the notion of parallel transport to obtain the Berry phases of
individual bands. We refer to Ref. [#Olsen]_ for details.

For materials with time-reversal symmetry the Chern number vanishes. Instead,
any insulator in two dimensions can be classified according to a
`\mathbb{Z}_2` index `\nu` that counts the number of times berry phases
acquire a particular value in half the Brillouin zone modulo two. Below we
give the example of stanene, which is referred to as a quantum spin Hall
insulator due to the non-trivial topology reflected by `\nu=1`. The ground
state can be set up with the script :download:`gs_Sn.py`. Afterwards the
Berry phases of all occupied bands are calculated with

.. literalinclude:: Sn_parallel_transport.py

Finally the berry phase spectrum can be plottet with
:download:`plot_phase.py` and the result is shown below.

.. image:: phases.png
    :height: 400 px

Note the degeneracy of all phases at the time-reversal invariant points
`\Gamma` and `M`. Also note that any horizontal line is transversed by an odd
number of phases in half the Brillouin zone (for example the `\Gamma-M`
line). We also display the expectation value of `S_z` according to color.
This is possible because the individual phases correspond to the first
moments of hybrid Wannier functions localized along the `x`-direction and
these functions have a spinorial structure with a well-defined value of
`\langle S_z\rangle`. [#Olsen]_

Polarization from from parallel transport
=========================================

The parallel transport module can also be used to calculate the polarization.
Strictly speaking the parallel transport algorithm is not required for the
polarization because it involves a trace over occupied bands. Nevertheless,
it is reassuring that we obtain the correct polarization from the parallel
transport. More importantly, the parallel transport supports spin-orbit
coupling and non-collinear magnetism, which the polarization module introduced
above does not. The script

.. literalinclude:: parallel_BaTiO3.py

calculates the polarization of BaTiO3 from the parallel transport module. Note
that the parallel transport only supports 2D `k` grids and we thus loop
over slices at fixed `k_y` to calculate the polarization. It is also important
to realize that the phases at individual points are only defined modulo `2\pi`
and the phases has to be chosen such that the bands are continuous. The script
is seen to exactly reproduce the value of 0.45 `C/m^2` found above. Spin-orbit
coupling may be switched on by setting scale=1 in the script.

.. [#Zhang] Y. Zhang, J. Sun, J. P. Perdew, and X. Wu
	    :doi:`10.1103/PhysRevB.96.035143`
.. [#King-Smith] R. D. King-Smith, D. Vanderbilt
	    :doi:`10.1103/PhysRevB.47.1651`
.. [#Olsen] T. Olsen, E. Andersen, T. Okugawa, D. Torelli, T. Deilmann, K. S. Thygesen
	    :doi:`10.1103/PhysRevMaterials.3.024005`
