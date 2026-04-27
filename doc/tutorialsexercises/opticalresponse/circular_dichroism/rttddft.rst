.. _circular_dichroism_rtddft:

============================================
Circular dichroism (CD) with real-time TDDFT
============================================

In this tutorial, we calculate the rotatory strength spectrum of
:download:`(R)-methyloxirane molecule <r-methyloxirane.xyz>`.

The equations underlying the implementation are described in
Ref. [#Makkonen2021]_.


LCAO mode
---------

In this example, we use :ref:`real-time TDDFT LCAO mode <lcaotddft>`.
We recall that the CD-LCAO calculations are in general
:ref:`sensitive to the used basis sets <note basis sets>`,
so we will demonstrate how to construct augmented basis sets,
and compare them to standard basis sets. Additionally, a choice of gauge
is essential. GPAW provides two possibilities, which are selected by providing
either ``gauge='length'`` (default) or ``gauge='velocity'`` keyword argument to
the ``absorption_kick``-method.

We augment the default dzp basis sets with
numerical Gaussian-type orbitals (NGTOs),
for details see :ref:`augmented basis set generation <aug in lcao>` :

.. literalinclude:: lcao/basis.py

Here, the Gaussian parameters correspond to diffuse functions
in aug-cc-pvdz basis sets as obtained from
`Basis Set Exchange <https://www.basissetexchange.org/>`_.

Let's do the ground-state calculation (note the ``basis`` keyword):

.. literalinclude:: lcao/gs.py

Then, we carry the time propagation as usual in
:ref:`real-time TDDFT LCAO mode <lcaotddft>`,
but we attach :class:`~gpaw.tddft.MagneticMomentWriter`
to record the time-dependent magnetic moment.
In this script, we wrap the time propagation code
inside ``main()`` function to make the same script reusable
with ``gauge='length' or 'velocity'`` and different kick directions:

.. literalinclude:: lcao/td.py

.. tip::

    When running the script with ``gpaw``-command, one must add ``--`` before the actual arguments to ensure they are correctly passed the script. For example: ``$ gpaw -P 4 python td.py -- --gauge=length --kick=x``.

After repeating the calculation for kicks in x, y, and z directions in length gauge and velocity gauge, respectively,
we calculate the rotatory strength spectrum from the magnetic moments:

.. literalinclude:: lcao/spec.py

Comparing the resulting spectrum to one calculated with plain dzp basis sets shows
the importance of augmented basis sets. For comparison of different gauge choices, see below
at the next section Origin dependence.

.. image:: lcao/spectra.png


Origin dependence
-----------------

Length gauge
~~~~~~~~~~~~

Circular dichroism spectra obtained in length gauge can exhibit pronounced origin dependence.
Below, this is demonstrated by plotting the circular dichroism spectra calculated in the length gauge using different origins. 
See the documentation of :class:`~gpaw.tddft.MagneticMomentWriter`
for parameters controlling the origin location.

The magnetic moment data can be written at multiple different origins
during a single propagation as demonstrated in this script:

.. literalinclude:: lcao/td_origins.py


Velocity gauge
~~~~~~~~~~~~~~

In the velocity gauge, circular dichroism spectra are less origin dependent since the velocity form of the dipole operator satisfies gauge invariance more naturally in comparison to its length gauge form. By reducing of the artificial dependence on the choice of coordinate origin during dipole moment calculations, the velocity gauge can provide more consistent and physically meaningful circular dichroism spectra across different origins.

In the following image, the length and velocity gauge spectra with different origins are compared. We see that the velocity gauge
has much smaller origin dependence:

.. image:: lcao/spectra_origins.png


FD mode
-------

In this example, we use :ref:`real-time TDDFT FD mode <timepropagation>`.

Let's do the ground-state calculation (note that FD mode typically requires
larger vacuum and smaller grid spacing than LCAO mode; convergence
with respect to these parameters need to be checked for both modes separately):

.. literalinclude:: fd/gs.py

Then, similarly to LCAO mode, we carry the time propagation as usual and attach
:class:`~gpaw.tddft.MagneticMomentWriter`
to record the time-dependent magnetic moment
(note the ``tolerance`` parameter for the iterative solver;
smaller values might be required when signal is weak):

.. literalinclude:: fd/td.py

After repeating the calculation for kicks in x, y, and z directions,
we calculate the rotatory strength spectrum from the magnetic moments:

.. literalinclude:: fd/spec.py

The resulting spectrum:

.. image:: fd/spectrum.png

The spectrum compares well with the one obtained
with LCAO mode and augmented basis sets.

References
----------

.. [#Makkonen2021]
   | E. Makkonen, T. P. Rossi, A. H. Larsen, O. Lopez-Acevedo, P. Rinke,  M. Kuisma, and X. Chen,
   | :doi:`Real-time time-dependent density functional theory implementation of electronic circular dichroism applied to nanoscale metal-organic clusters <10.1063/5.0038904>`
   | J. Chem. Phys. **154**, 114102 (2021)
