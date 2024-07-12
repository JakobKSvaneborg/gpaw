.. _sites:

=============================================
Local properties of individual magnetic sites
=============================================

It is almost always very useful to analyze magnetic systems in terms of the
individual magnetic sites of the crystal. In this tutorial, we illustrate how
to calculate individual site properties for the magnetic atoms in GPAW.

Since it is not well-defined *a priori* where one site ends and another begins,
GPAW supplies functionality to calculate the site properties as a function of
spherical radii `r_\mathrm{c}`. In this picture, the site properties are defined
in terms of integrals with unit step functions
`\Theta(\mathbf{r}\in\Omega_{a})`, which are nonzero only inside a sphere of
radius `r_\mathrm{c}` around the given magnetic atom `a`.

Local functionals of the spin-density
=====================================

.. module:: gpaw.response.site_data

For any functional of the (spin-)density `f[n, \mathbf{m}](\mathbf{r})`,
one may define a corresponding site quantity,

.. math::
   f_a = \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a})
   f[n,\mathbf{m}](\mathbf{r}).

GPAW supplies functionality to compute such site quantities defined based on
*local* functionals of the spin-density for collinear systems,
`f[n,\mathbf{m}](\mathbf{r}) = f(n(\mathbf{r}),n^z(\mathbf{r}))`.
The implementation (using the PAW method) is documented in [#Skovhus]_.

In particular, the site magnetization,

.. math::
   m_a = \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a}) n^z(\mathbf{r}),

can be calculated via the function :func:`calculate_site_magnetization`, whereas
the function :func:`calculate_site_zeeman_energy` computes the LSDA site Zeeman
energy,

.. math::
   E_a^\mathrm{Z} = - \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a})
   W_\mathrm{xc}^z(\mathbf{r}) n^z(\mathbf{r}).

Example: Iron
-------------

In the script
:download:`Fe_site_properties.py`,
the site magnetization and Zeeman energy are calculated from the ground state
of bcc iron. The script should take less than 10 minutes on a 40 core node.
After running the calculation script, you can download and excecute
:download:`Fe_plot_site_properties.py`
to plot the site magnetization and Zeeman energy as a function of the
spherical site radius `r_\mathrm{c}`.

.. image:: Fe_site_properties.png
	   :align: center

Although there does not exist an *a priori* magnetic site radius `r_\mathrm{c}`,
we clearly see that there is a region, where the site Zeeman energy is constant
as a function of the radius, hence making `E_a^\mathrm{Z}` a well-defined
property of the system in its own right.
However, the same cannot be said for the site magnetization, which continues to
varry as a function of the cutoff radius. This is due to the fact that the
interstitial region between the Fe atoms is slightly spin-polarized
anti-parallel to the local magnetic moments, resulting in a radius
`r_\mathrm{c}^\mathrm{max}` which maximizes the site magnetization (marked with
a dotted line). If one wants to employ a rigid spin approximation for the
magnetic site, i.e. to assume that the direction of magnetization is constant
within the site volume, it is natural to choose `r_\mathrm{c}^\mathrm{max}` to
define the sites. In practice, `r_\mathrm{c}^\mathrm{max}` can be calculated
(along with the maximized magnetic moment) via the function
:func:`maximize_site_magnetization` and in general, the allowed ranges of atomic
cutoff radii can be inspected via the :func:`get_site_radii_range` function.


Site-based sum rules
====================

.. module:: gpaw.response.mft

In addition to site quantities, one may also introduce the concept of site
matrix elements, that is, expectation values of functionals
`f(\mathbf{r})=f[n, \mathbf{m}](\mathbf{r})`
evaluated on specific spherical sites,

.. math::
   f^a_{n\mathbf{k}s,m\mathbf{k}+\mathbf{q}s'} = \langle \psi_{n\mathbf{k}s}|
   \Theta(\mathbf{r}\in\Omega_{a}) f(\mathbf{r})
   |\psi_{m\mathbf{k}+\mathbf{q}s'} \rangle
   = \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a}) f(\mathbf{r})\,
   \psi_{n\mathbf{k}s}^*(\mathbf{r})
   \psi_{m\mathbf{k}+\mathbf{q}s'}(\mathbf{r}).

Similar to the site quantities, GPAW includes functionality to calculate site
matrix elements for arbitrary *local* functionals of the (spin-)density
`f(\mathbf{r}) = f(n(\mathbf{r}),n^z(\mathbf{r}))`, with implementational
details documented in [#Skovhus]_.
For example, one can calculate the site pair density

.. math::
   n^a_{n\mathbf{k}s,m\mathbf{k}+\mathbf{q}s'} =
   \langle \psi_{n\mathbf{k}s}|
   \Theta(\mathbf{r}\in\Omega_{a})
   |\psi_{m\mathbf{k}+\mathbf{q}s'} \rangle

as well as the site Zeeman pair energy

.. math::
   E^{\mathrm{Z},a}_{n\mathbf{k}s,m\mathbf{k}+\mathbf{q}s'}=-
   \langle \psi_{n\mathbf{k}s}|
   \Theta(\mathbf{r}\in\Omega_{a}) W_\mathrm{xc}^z(\mathbf{r})
   |\psi_{m\mathbf{k}+\mathbf{q}s'} \rangle.


Now, from such site matrix elements, one can formulate a series of sum rules for
various site quantities. For instance, one can construct single-particle sum
rules for both the site magnetization and the site Zeeman energy, simply by
summing over the diagonal of the site matrix elements for all the occupied
states, weighted by the Pauli matrix `\sigma^z`,

.. math::
   m_a = \frac{1}{N_k} \sum_\mathbf{k} \sum_{n,s}
   \sigma^z_{ss} f_{n\mathbf{k}s} n^a_{n\mathbf{k}s,n\mathbf{k}s},

.. math::
   E_a^\mathrm{Z} = \frac{1}{N_k} \sum_\mathbf{k} \sum_{n,s}
   \sigma^z_{ss} f_{n\mathbf{k}s}
   E^{\mathrm{Z},a}_{n\mathbf{k}s,n\mathbf{k}s}.

Although trivial, these sum rules can be used as a consistency tests for the
implementation and can be accessed via the functions
:func:`calculate_single_particle_site_magnetization`
and
:func:`calculate_single_particle_site_zeeman_energy`.

In addition to the single-particle sum rules, one may also introduce actual
pair functions that characterize the band transitions of the system.
In particular, one may introduce the so-called pair site magnetization

.. math::
   m_{ab}(\mathbf{q}) = \frac{1}{N_k} \sum_\mathbf{k} \sum_{n,m}
   \left( f_{n\mathbf{k}\uparrow} - f_{m\mathbf{k}+\mathbf{q}\downarrow} \right)
   n^a_{n\mathbf{k}\uparrow,m\mathbf{k}+\mathbf{q}\downarrow}
   n^b_{m\mathbf{k}+\mathbf{q}\downarrow,n\mathbf{k}\uparrow}

and pair site Zeeman energy

.. math::
   E^\mathrm{Z}_{ab}(\mathbf{q}) = \frac{1}{N_k} \sum_\mathbf{k} \sum_{n,m}
   \left( f_{n\mathbf{k}\uparrow} - f_{m\mathbf{k}+\mathbf{q}\downarrow} \right)
   E^{\mathrm{Z},a}_{n\mathbf{k}\uparrow,m\mathbf{k}+\mathbf{q}\downarrow}
   n^b_{m\mathbf{k}+\mathbf{q}\downarrow,n\mathbf{k}\uparrow},

which turn out to be `\mathbf{q}`-independent diagonal pair functions,
`m_{ab}(\mathbf{q})=\delta_{ab} m_a` and
`E^{\mathrm{Z}}_{ab}(\mathbf{q})=\delta_{ab} E^\mathrm{Z}_a`,
thanks to a simple sum rule [#Skovhus]_. Because the sum rule relies on the
completeness of the Kohn-Sham eigenstates, it breaks down when using only a
finite number of bands. Hence, it can be useful to study the band convergence of
`m_{ab}(\mathbf{q})` and `E^{\mathrm{Z}}_{ab}(\mathbf{q})` to gain insight
about related completeness issues of more complicated pair functions. In GPAW,
they can be calculated using the :func:`calculate_pair_site_magnetization` and
:func:`calculate_pair_site_zeeman_energy` functions.


Example: Iron
-------------

In the
:download:`Fe_site_sum_rules.py`
script, the single-particle site Zeeman energy is calculated along with the
pair site Zeeman energy using a varrying number of bands. It should take less
than half an hour on a 40 core node to run.
Having done so, you can excecute
:download:`Fe_plot_site_sum_rules.py`
to plot the band convergence of `E^{\mathrm{Z}}_{ab}(\mathbf{q})`.

.. image:: Fe_site_sum_rules.png
	   :align: center

Whereas the single-particle site Zeeman energy (dotted line) is virtually
identical to the Zeeman energy calculated from the spin-density (blue line),
there are significant deviations from the two-particle site Zeeman energy sum
rule, especially with a low number of bands.
Including at least 12 bands beyond the *4s* and *3d* valence bands, we obtain a
reasonable fulfillment of the sum rule in the region of radii, where the site
Zeeman energy is flat. Interestingly, this is not the case at smaller site
radii, meaning that the remaining incompleteness shifts the site Zeeman energy
away from the nucleus, while remaining approximately constant when integrating
out the entire augmentation sphere.

In the figure, we have left out the imaginary part of the pair site Zeeman
energy. You can check yourself that it vanishes more or less identically.

Exchange parameters
===================

Bla bla bla, see also :ref:`mft`

Example: Cobalt
---------------

In the
:download:`Co_exchange_parameters.py`
script, we calculate the Co exchange parameters `\bar{J}^{ab}(\mathbf{q})` as
a function of the cutoff radius `r_\mathrm{c}` for the `\Gamma`, M, K and A
high-symmetry points as well as the exchange parameters at the ideal rigid spin
approximation cutoff `r_\mathrm{c}^\mathrm{max}` for all commensurate q-points
along the corresponding `\Gamma`-M-K-`\Gamma`-A high-symmetry path. It should
take less than an hour on a 40 core node to run.
You can then excecute
:download:`Co_plot_hsp_magnons_vs_rc.py`
to examine the `r_\mathrm{c}`-dependence of the exchange parameters via the
resulting high-symmetry point magnon energies.

.. image:: Co_hsp_magnons_vs_rc.png
	   :align: center

Once again, there is a region of cutoff radii `r_\mathrm{c}` in which the magnon
dispersion (site magnetization is held constant) is more or less independent of
`r_\mathrm{c}`. Thus, although there does not exist an *a priori* concept of a
Co magnetic site, we see that the isotropic exchange couplings
`\bar{J}^{ab}(\mathbf{q})` resulting from a projection of the exchange field
onto spherical sites itself is well-defined.

Excecuting
:download:`Co_plot_dispersion.py`,
you can explore the full magnon dispersion of hcp-Co as calculated within LDA in
the LR-MFT method.

.. image:: Co_dispersion.png
	   :align: center


Excercises
==========

To get comfortable with the presented functionality, here are some suggested
excercises to get you started:

1) Calculate the site pair magnetization of iron and analyze its band
   convergence.

2) Investigate the sensitivity of the site pair functions as a function of the
   wave vector `\mathbf{q}`.

3) Calculate the site magnetization and spin splitting for a ferromagnetic
   material with inequivalent magnetic sublattices.

   a) Are you still able to find ranges of radii, where the site Zeeman energy
      is constant?
   b) What happens to the band convergence of the pair functions?
   c) How does the off-diagonal elements of the pair functions converge as a
      function of the number of bands?


API
===

.. module:: gpaw.response.site_data
.. autofunction:: calculate_site_magnetization
.. autofunction:: calculate_site_zeeman_energy
.. autofunction:: maximize_site_magnetization
.. autofunction:: get_site_radii_range
.. module:: gpaw.response.mft
.. autofunction:: calculate_single_particle_site_magnetization
.. autofunction:: calculate_single_particle_site_zeeman_energy
.. autofunction:: calculate_pair_site_magnetization
.. autofunction:: calculate_pair_site_zeeman_energy


References
==========

.. [#Skovhus] T. Skovhus and T. Olsen,
	   *publication in preparation*, (2024)
