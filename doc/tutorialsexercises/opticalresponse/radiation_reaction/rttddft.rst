.. _radiation_reaction_rttddft:

=================================================================================
Radiation-Reaction: Self-consistent Light-Matter interaction with real-time TDDFT
=================================================================================

In this tutorial, we calculate the superradiant absorption-spectrum of a sodium-dimer chain.

The equations underlying the implementation are described in
Ref. [#Schaefer2021]_. The here used parameters are coarser than the published work.
This implementation is so far only available for the LCAO mode.

We use :ref:`real-time TDDFT LCAO mode <lcaotddft>`
and add the radiation-reaction potential for emission into a waveguide.
The available implementation adds a local potential that accounts for the recoil
that emerges as a consequence of the emission of light. This approach effectively
embeds the electromagnetic environment into the KS equation and ensures a
self-consistent interaction between light and matter.
The keyword ``RRemission`` demands the cross-sectional area of the waveguide
and the polarization of the propagating electric field.
With decreasing cross-sectional area, the emission will be stronger.
Thus far, 1d-waveguide emission is the only available electromagnetic environment
but further additions are scheduled.

We will start by calculating the emission spectrum in z-direction for a single sodium dimer:

.. literalinclude:: lcao_linres_NaD.py

The polarization of the electric field in the waveguide is aligned with the z-axis, exerting
a recoil-force on the electronic coordinates that oscillate in this direction.
As a result, the electronic motion is damped as energy is emitted into the waveguide, in line
with Newtons third law.

Let us next add a second dimer parallel to the first one and orthogonal to the chain-axis
(H-aggregate configuration). For reasons that will become clear in the following, we can get
away with half the propagation time.

.. literalinclude:: lcao_linres_NaD2chain.py

We smoothed the spectra in both cases with a sharp Lorentzian for visual appearance but the observed width
corresponds to the correct lifetimes. In the perturbative limit, this corresponds to Wigner-Weisskop theory.
Two features can be observed.

.. image:: spectra_nad.png

1) The width of two sodium dimers is double the width of a single one.
For a set of non-interacting matter-subsystems, their emission probability of a single photon
scales linearly with the number of subsystem, i.e., the rate increases here linearly with the number
of dimers. Intuitively, more charge oscillating provides a larger dipole which leads to a stronger emission.
Clearly, the dimers are not entirely non-interacting for distance of 0.8 nm,
which leads to the following observation.

2) The peak is shifted slightly to higher frequencies.
The parallel configuration between the dimers results in a coupling between the dipolar excitations that shift
the excitation energy to slightly higher values. This configuration is also known as H-aggregate.
A head to end configuration on the other hand is known as J-aggregate, providing a red-shift.

It is important to realize here that the propagation time has to be long enough to allow the
recoil-forces to substantially damp the oscillations, i.e., enough time is needed to capture the decay.

.. image:: dipoles.png

Often this might be unfeasible or unsuitable such that an extrapolation from small cross-sectional areas
and quick decay to larger areas and slower decay might be beneficial.
A generalized implementation, including strong coupling and complex electromagnetic environments is
currently in develop.

References
----------

.. [#Schaefer2021]
   | C. Schaefer, and G. Johansson,
   | :doi:`A Shortcut to Self-Consistent Light-Matter Interaction and Realistic Spectra from First-Principles <10.1103/PhysRevLett.128.156402>`
   |  Phys. Rev. Lett. 128, 156402 (2022)
