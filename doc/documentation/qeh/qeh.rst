=============================================================
The Quantum Electrostatic Heterostructure (QEH) model: Theory
=============================================================

We follow the notation of Ref [#andersen2015]_.
For each monolayer in a heterostructure, the monolayer response function
`\widetilde{\chi}(\mathbf{r}, \mathbf{r}', q_\parallel, \omega)` is
first calculated. We assume here that `\widetilde{\chi}` is
isotropic, i.e. only a function of
`q_\parallel = |\mathbf{q}_\parallel|`, and independent of the
direction of `\mathbf{q}_\parallel`. The response function is averaged
over the in-plane coordinates, and we define

.. math::
   :label: eq_chizz

   	\widetilde{\chi}(z, z', q_\parallel, \omega) = \frac{1}{A} \int_A \int_A \mathrm{d} \mathbf{{r}}_\parallel \mathrm{d} \mathbf{{r}}'_\parallel \widetilde{\chi}(\mathbf{r}, \mathbf{r}', q_\parallel, \omega),

where the integration is over the in-plane coordinates, and `A` is
the in-plane area of the supercell. The `z`-dependence can be
approximated in a monopole-dipole basis, in which we express
`\widetilde{\chi}` as a `2 \times 2` matrix
`\chi_{\alpha \alpha'}`, where `\alpha=0` corresponds to a
monopole component, while `\alpha = 1` corresponds to a dipole
component, and likewise for `\alpha'`. These components are given
by

.. math:: 
   :label: eq_chi_alpha

      \widetilde{\chi}_{\alpha \alpha'} (q_\parallel, \omega) = \int \int \mathrm{d}z \mathrm{d}z'  (z-z_c)^\alpha  \widetilde{\chi}(z, z', q_\parallel, \omega) (z'-z_c)^{\alpha'},

where each integral runs over the interval
`[z_c - \frac{L}{2}, z_c + \frac{L}{2}]`, where `L` is the
thickness of the layer, and `z_c` the position of the middle of
the layer. To make explicit the monopole/dipole structure, we label the
components of the `\chi_{\alpha \alpha'}` matrix as
`\alpha \in {M, D}`, where `M` corresponds to
`\alpha=0` and `D` to `\alpha = 1.` This corresponds
to the naming convention used in the GPAW implementation.

Expressed in a plane-wave basis, we have

.. math::

   \widetilde{\chi}(\mathbf{r}, \mathbf{r}', q_\parallel, \omega) 
   	= \frac{1}{\Omega} \sum_{\mathbf{G} \mathbf{G}'} e^{i(\mathbf{q}_\parallel + \mathbf{G})\cdot \mathbf{r}} \widetilde{\chi}_{\mathbf{G}\mathbf{G}'}(q_\parallel, \omega) e^{-i(\mathbf{q}_\parallel + \mathbf{G'})\cdot \mathbf{r}'},

`\Omega` being the volume of the supercell. Integrating over the
plane corresponds to taking
`\mathbf{G}_\parallel = \mathbf{G}_\parallel' = 0`, such that equation
:eq:`eq_chizz` becomes

.. math::

   \widetilde{\chi}(z, z', q_\parallel, \omega) = \frac{1}{L} 
   	 \sum_{G_z G_z'} e^{iG_z z} \widetilde{\chi}_{G_z G_z'}(q_\parallel, \omega) e^{-iG_z' z'}

The integrals over `z` in equation :eq:`eq_chi_alpha` can then be carried out analytically, and
we find

.. math::

   \begin{aligned}
   	 &\widetilde{\chi}_M(q_\parallel, \omega) = L \widetilde{\chi}_{G_z = 0, G_z' = 0} \\
   	 &\widetilde{\chi}_{MD}(q_\parallel, \omega) = \sum_{G_z' \neq 0} \widetilde{\chi}_{0,G_z'} z_F^*(G_z') \\
   	 &\widetilde{\chi}_{DM}(q_\parallel, \omega) = \sum_{G_z \neq 0} z_F(G_z) \widetilde{\chi}_{G_z,0}  \\
   	 &\widetilde{\chi}_{D}(q_\parallel, \omega) = \frac{1}{L} \sum_{G_z \neq 0, G_z' \neq 0} z_F(G_z) \widetilde{\chi}_{G_z G_z'}z_F^*(G_z'),
    \end{aligned}

where the so-called *z-factor* `z_F` is

.. math::

   z_F(G_z) = \int_{z_c - \frac{L}{2}}^{z_c + \frac{L}{2}} e^{i G_z z} z \mathrm{d}z   	= -\frac{i e^{i G_z z_c}}{G_z^2} \left[G_z L \cos\left(\frac{G_z L}{2}\right) - 2 \sin\left(\frac{G_z L}{2}\right)\right],

and `z_F^*` is the complex conjugate of `z_F`.

For systems with mirror symmetry in the out of plane (`z`)
direction, the off-diagonal elements `\chi_{MD}` and
`\chi_{DM}` must vanish. This can be seen from the following: the mirror symmetry implies that
`\chi(z,z') = \chi(-z, -z')`, where we have set `z_c = 0`
for simplicity, and we have then for e.g. `\chi_{DM}` that

.. math::

   \chi_{DM} = \int  z \chi(z,z') \mathrm{d}z\mathrm{d}z'
    = \int  z \chi(-z,-z') \mathrm{d}z \mathrm{d}z'
    = \int  (-z) \chi(z,z') \mathrm{d}z \mathrm{d}z'
    = - \chi_{DM}

where for the last equality we made the substitution
`z \rightarrow-z` and `z' \rightarrow- z'`. A similar result
holds for `\chi_{MD}`. Therefore one only needs to calculate the off-diagonal elements for materials that do not have mirror symmetry. 


.. [#andersen2015] Andersen, Kirsten, Simone Latini, and Kristian S. Thygesen.
            Dielectric genome of van der Waals heterostructures,
            *Nano letters* 15.7 (2015): 4616-4621.
            