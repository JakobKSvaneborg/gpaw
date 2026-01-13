.. _raman:

========================
Raman spectroscopy
========================

GPAW offers two ways of calculating Raman intensities. One can use the `ASE Raman
<https://ase-lib.org/ase/vibrations/raman.html>`_ utility together
with the GPAW LRTDDFT module as shown in the Resonant Raman tutorial :ref:`resonant_raman_water`.

GPAW also implements Raman spectroscopy for zone-center phonons of extended systems
using the electron-phonon coupling (see :ref:`elphtheory` and :ref:`elph`) within 3rd order perturbation
theory :dfn:`Taghizadeh et a.` [#Taghizadeh2020]_ , which is discussed here. This method is currently only
implemented for the LCAO mode.

The Stokes Raman intensity can be written as

.. math::

    I(\omega) = I_0 \sum_\nu \frac{n_\nu+1}{\omega_\nu} \vert
    \sum_{\alpha, \beta} u_{in}^\alpha R_{\alpha \beta}^\nu u_{out}^\beta
    \vert^2 \delta(\omega-\omega_\nu)

where `\nu` denotes phonon modes and `\alpha`, `\beta` denote polarizations
of the incoming and outgoing laser light.
The Raman tensor `R_{\alpha \beta}^\nu` has six terms and is given by
Ref. [#Taghizadeh2020]_ Eq. (10)

.. math::

    R_{\alpha \beta}^\nu \equiv \sum_{ijmn \mathbf{k}} \left[
    \frac{p_{ij}^\alpha (g_{jm}^\nu \delta_{in} - g_{ni}^\nu \delta_{jm})p_{mn}^\beta}{(\hbar \omega_{in}-\varepsilon_{ji})(\hbar \omega_{out}-\varepsilon_{mn})} +
    \frac{p_{ij}^\alpha (p_{jm}^\beta \delta_{in} - p_{ni}^\beta \delta_{jm})g_{mn}^\nu}{(\hbar \omega_{in}-\varepsilon_{ji})(\hbar \omega_{\nu}-\varepsilon_{mn})} + \\
    \frac{p_{ij}^\beta (g_{jm}^\nu \delta_{in} - g_{ni}^\nu \delta_{jm})p_{mn}^\alpha}{(-\hbar \omega_{out}-\varepsilon_{ji})(-\hbar \omega_{in}-\varepsilon_{mn})} +
    \frac{p_{ij}^\beta (p_{jm}^\alpha \delta_{in} - p_{ni}^\alpha \delta_{jm})g_{mn}^\nu}{(-\hbar \omega_{out}-\varepsilon_{ji})(\hbar \omega_{\nu}-\varepsilon_{mn})} + \\
    \frac{g_{ij}^\nu (p_{jm}^\alpha \delta_{in} - p_{ni}^\alpha \delta_{jm})p_{mn}^\beta}{(-\hbar \omega_{\nu}-\varepsilon_{ji})(\hbar \omega_{out}-\varepsilon_{mn})} +
    \frac{g_{ij}^\nu (p_{jm}^\beta \delta_{in} - p_{ni}^\beta \delta_{jm})p_{mn}^\alpha}{(-\hbar \omega_{\nu}-\varepsilon_{ji})(-\hbar \omega_{in}-\varepsilon_{mn})}
    \right] f_i(1-f_j)f_n(1-f_m)

The first term is considered to be the resonant term of the expression, the other
terms represent different time orderings of the interaction in the Feynman diagrams.

To compute the Raman intensity we need these ingredients: The momentum matrix
elements `p_{ij}^\alpha=\langle i \mathbf{k} | \hat p^\alpha| j \mathbf{k} \rangle`,
the electron-phonon matrix `g_{ij}^\nu = \langle i \mathbf{k} \vert \partial_{\nu{q=0}} V^{KS} \vert j \mathbf{k} \rangle`
in the optical limit `\mathbf{q}=0` and of course knowledge of the electronic
states and phonon modes throughout the Brillouin zone.
For these calculations we can employ in the :meth:`~gpaw.lcao.dipoletransition.get_momentum_transitions`
method, the GPAW electron-phonon module :ref:`elph` and the ASE phonon module, respectively.

It is required to calculate all properties for the full Brillouin zone using `symmetry=off`. By default the routine saves a file called ``mom_skvnm.npy`` containing the
momentum matrix. This can be deactivated using the ``savetofile`` switch. The
matrix is always the return value of :meth:`~gpaw.lcao.dipoletransition.get_momentum_transitions`.

Energy changes for phonons and potential changes for electron-phonon couplings
are both computed using a finite displacement technique. Both quantities can be
obtained simultaneously. In principle the phonon modes can be obtained in any
fashion, which yields an ASE phonon object though.
For small systems the finite displacement method has the disadvantage of leading
to an interaction of a displaced atom with its periodic images. This can lead to
large errors especially in the electron-phonon matrix. This can be avoided by using
a sufficiently large supercell for the finite displacement simulations.

If phonon and effective potential are calculated simultaneously, results are saved
in the same cache files with default name `elph`.

Some more details are elaborated in the related tutorial :ref:`elphraman`.

----------
References
----------

.. [#Taghizadeh2020] A. Taghizadeh, U. Leffers, T.G. Pedersen, K.S. Thygesen,
                   "A library of ab initio Raman spectra for automated
                   identification of 2D materials",
                   *Nature Communications* **11**, 3011 (2020).

----
Code
----

.. autoclass:: gpaw.elph.ResonantRamanCalculator
    :members:

.. autoclass:: gpaw.elph.RamanData
    :members:

.. autofunction:: gpaw.lcao.dipoletransition.get_momentum_transitions
