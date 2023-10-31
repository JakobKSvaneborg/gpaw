.. module:: gpaw.new.spinspiral
.. _spinspiral calculations:

========================
Spin spiral calculations
========================

In this tutorial we employ the Generalized Bloch's Theorem (gBT) approach to
calculate the spin spiral ground-state [#Knöpfle]_. In this approach we can
choose any wave vector of the spin spiral :math:`\mathbf{q}`, and rotate the spin degrees
through the periodic boundary conditions accordingly. This rotation can be
included in Bloch's theorem by applying a combined translation and spin
rotation to the wave function at the boundaries. Then we get the generalized
Bloch's theorem,

.. math::

    \varphi_{\mathbf{q},\mathbf{k}}(\mathbf{r})=
    e^{i\mathbf{k}\cdot\mathbf{r}} U_\mathbf{q}^{\dagger}(\mathbf{r})
    \begin{pmatrix}
    u^{\uparrow}_{\mathbf{q},\mathbf{k}}(\mathbf{r})\\
    u^{\downarrow}_{\mathbf{q},\mathbf{k}}(\mathbf{r}),
    \end{pmatrix}

where `u^{\uparrow}_{\mathbf{q},\mathbf{k}}(\mathbf{r})` and 
`u^{\downarrow}_{\mathbf{q},\mathbf{k}}(\mathbf{r})` are spin dependent and
periodic Bloch functions modulated by the spin rotation matrix 

.. math::
   U_\mathbf{q}(\mathbf{r})=
   \begin{pmatrix}
   e^{i\mathbf{q}\cdot\mathbf{r}/2} & 0\\
   0 & e^{-i\mathbf{q}\cdot\mathbf{r}/2}
   \end{pmatrix}

which act as an enveloping function together with the standard Bloch envelope
`e^{i\mathbf{k}\cdot\mathbf{r}}`. This wavefunction ansatz make the magnetic
field rotate around the z-axis while leaving the density unchanged as

.. math::

   \hat{\mathbf{B}} = [\cos(\mathbf{q} \cdot \mathbf{a}),
                       \sin(\mathbf{q} \cdot \mathbf{a}),
                       0]^T

where `\mathbf{a}` are lattice translations. The magnetization direction follows
the direction of the magnetic field when the LDA exchange-functional is used.
These are called flat spin spirals. There is nothing special about the `xy`-plane, 
used here, since spin-orbit is neglected at this stage, the spin spiral is
invariant under any global spinor rotation. The reward is that we can simulate
any incommensurate spin spiral of this type in the chemical unit cell.

The initial magnetization is a input parameter, and when the gBT is applied the
magnetic moments should correspond to the non-periodic real space structure.
As an example consider making a spin spiral calculation in a 1D chain. The
spin structure with order parameter `q = [0.25, 0, 0]` requires a 4
times enlarged supercell. The input magnetic moments at each site will be rotated
by 90 degrees such that `\mathbf{m}_1=[m, 0, 0], \mathbf{m}_2=[0, m, 0],
\mathbf{m}_3=[-m, 0, 0], \mathbf{m}_1=[0, -m, 0]`. The corresponding spin spiral
calculation could be done in a calculation with only one magnetic atom, however
for illustrative purposes, suppose that we have a unit cell with two atoms then
the magnetic moments initialized to `\mathbf{m}_1=[m, 0, 0], \mathbf{m}_2=[0, m, 0]`.
(Note that the alternative would have been to initalize them in parallel, and then
rely on :math:`U_\mathbf{q}(\mathbf{r})` to do the rotation, however this is not
applied to the input moments because :math:`U_\mathbf{q}(\mathbf{r})` is absorbed
into the projector overlap functions). This magnetization is expected to be found
during the self-consistent cycle, however choosing it as the initial magnetic
moments can significantly improve convergence.

The z-component being zero is not strictly enforced. Thus, convergence to conical
spin spirals is in principle possible if initialized with components both in the
`xy` plane and in the `z` direction. However, typically the magnetization will converge
to a collinear state in along the z-axis or the flat spiral in the
`xy`-plane, depending on the initial angle provided.

There are some limitations associated with these wave functions, because we 
assume spin structure to decouple from the lattice such that the density matrix
is invariant under the spin rotation. In order for this to be the case, we
can only apply spin orbit coupling perturbatively [#Sandratskii]_, and not as
part of the self consistent calculation. Furthermore, with the density being
invariant under the this spinor rotation, so will also the z-component of the
magnetization. This can be understood by looking at the  magnetization
density `\tilde{\rho} = I_2\rho + \sigma\cdot m` under the spin spiral
rotation, where one sees that the entire diagonal is left invariant. Thus
we are limited to spiral structures which have magnetization vectors


Ground state of Monolayer :math:`\text{NiI}_2`
==============================================

The nickel halide :math:`\text{NiI}_2` is a van der waals bonded crystal, where
each layers is a 1T monolayer. Thus the nickel form a triangular lattice, which
is an interesting platform because magnetic frustration. Particularly one finds
that multiple competing interactions in the heisenberg picture can the cause for
stabilizing an incommensurate spin spiral ground state :math:`q \approx 1/7` that
is found in experiments. We can predict using DFT calculations, however if one
were to use a supercell approach one would have to compare energies of many large
supercells approximating the wavelength. Here we scan the spin spiral groundstate
in a systematic way along the high-symmetry bandpath of the crystal. This is done
using the following script.


.. literalinclude:: nii2_sgs.py


As a results we find that two minima can be found along both orthogonal directions,
indeed one can find paths in the :math:`\mathbf{q}`-space and find the minima
are seperated by a very small barrier of about 2meV. 


.. figure:: e-spiral.png

   (see :download:`plot.py`)


The spin spiral groundstate breaks a lot of the symmetries in the crystal
spacegroup, for example :math:`\text{NiI}_2` is a centrosymmetric crystal
but the inversion symmetry is broken by the spin spiral order. The electronic
density can then polarize as a consequence of the magnetic order, which is
denoted a type II multiferroric material. Although some mirror symmetries
might persist, depending on the orientation of the spin spiral plane. The 
plane orientation is usually the normal plane vector 
:math:`\mathbf{n} = (\theta, \varphi)`. The normal plane vector can be 
determined self-consistently in a supercell calculation, however converging
exact angles can be very tricky. Instead we can leverege the small spin
spiral groundstate, do a scan of scan of :math:`\mathbf{n}`
non-self-consistently using the projected spin-orbit approximation
[#Sandratskii]_. In the following script we do such a scan, but limited to
the upper half hemisphere of points, since the lower is related by
time-reversal symmetry.

.. literalinclude:: nii2_soc.py

In order to plot the scan of the spherical surface, we choose here to do a
stereographic projection of the half-sphere, which puts the out-of-plane
direction `z` in the center of a circle. The radial coordinate of the circle
corresponds to the `\theta` direction and the angular coordinate to the `\phi`
direction. We find that the spin-orbit energy landscape has a hard axis and a
very flat easy plane as seen below. Besides the contour displayed on the
colorbar, we also show the points within :math:`50\mu eV` and :math:`1\mu eV`
of the minimum in midnight blue and black respectively. The z-axis is shown as
a black dot in the middle of the stereographic plot, and the white dot is the
minimum which corresponds to :math:`\mathbf{n} = (33, 301)` degrees with an
uncertainty of 1 degree.

.. figure:: soc.png

   (see :download:`plot_soc.py`)


.. [#Knöpfle] K. Knöpfle, L. M. Sandratskii, and J. Kübler
   Spin spiral ground state of γ-iron,
   Phys. Rev. B 62, 5564 – Published 1 September 2000
   :doi:`10.1103/PhysRevB.62.5564`

.. [#Sandratskii] L. M. Sandratskii,
   Insight into the Dzyaloshinskii-Moriya interaction through
   first-principles study of chiral magnetic structures
   Phys. Rev. B. 96. 024450 – Published 31 July 2017
   :doi:`10.1103/PhysRevB.96.024450`
