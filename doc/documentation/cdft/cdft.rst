.. module:: gpaw.cdft
.. _constrained_DFT:

======================
Constrained DFT (cDFT)
======================

Introduction
============

cDFT is a method for build charge/spin localized or diabatic states with a
user-defined charge and/or spin state. As such cDFT, is a useful tool for
widening the scope of ground-state DFT to excitation processes, correcting for
self- interaction energy in current DFT functionals, excitation energy,
and electron transfer as well as parametrizing model Hamiltonians, for example.


Theoretical Background
======================

The general cDFT methodology is reviewed in [#cdft1]_ and the publication
of the GPAW implementation is available in [#cdft2]_. In short, the cDFT works
by specifying an additional constraining term to the KS functional. The role of
this constraint is to enforce a user specified charge/spin state (`N_c`) on the
chosen regions of atoms. The constrained regions are specified by a weight
function `w(\mathbf{r})` and the strength of the constraining potential acting
on the specified region is `V_c`. With these definitions the new energy
functional with the constraint is

.. math:: F[n(\mathbf{r}), V_c] = E^{KS}[n] +
    \sum_c  V_c \sum_{\sigma}\left(\int d\mathbf{r}w_c^{\sigma}(\mathbf{r})
                                   n^{\sigma}(\mathbf{r})-N_c\right)

where `E^{KS}` is the Kohn-Sham energy, `c` specifies the region, and `\sigma`
is the spin variable. It is also seen that `V_c` is also a Lagrange multiplier.
The modified energy functional leads to an additional external potential

.. math:: v_{\rm eff}^{\sigma}=\dfrac{\delta E^{KS}[n(\mathbf{r})]}
    {\delta n(\mathbf{r})} + \sum_c  V_cw_c^{\sigma}(\mathbf{r})

This is just a sum of the usual KS potential and the constraining potential
which is also used in the self-consistent calculation.
The constraint is further enforced by introducing the convergence criteria

.. math:: C \geq \bigg\lvert \sum_{\sigma}
    \int {\rm d}\mathbf{r} w_c^{\sigma}(\mathbf{r})n^{\sigma}(\mathbf{r}) -
    N_c \bigg\rvert \, ,\forall\, c

The `V_c` is self-consistently optimized so that the specified constraints are
satisfied. Formally, this is written as

.. math:: F\left[V_{c}\right]=
    \min _{n} \max _{\left\{V_{c}\right\}}
    \left[E^{\mathrm{KS}}[n(\mathbf{r})]+
          \sum_{c} V_{c}\left(\int \mathrm{d} \mathbf{r} w_{c}(\mathbf{r})
                              n(\mathbf{r})-N_{c}\right)\right]

`V_c` is obtained from

.. math:: \frac{\mathrm{d} F}{\mathrm{d} V_{c}} =
    \int \mathrm{d} \mathbf{r} w_{c}(\mathbf{r}) n(\mathbf{r})-N_{c}=0

In the end, one ends up with a modified electron/spin density localized
on the chosen atoms.


Notes on using cDFT
===================

The weight function
-------------------

In the GPAW implementation a Hirschfeld partition scheme with atom-centered
Gaussian functions is used. These Gaussian have two tunable parameters: the
cut-off `R_c` and the width `\mu`. If the constrained region cuts a covalent
bond, the results are sensitive to width parameter. There is no universal
guide for choosing the width in such cases. A sensible choice is to compute
match the Gaussian-Hirschfeld charges with e.g. Bader charges. The function
:meth:`~gpaw.cdft.cdft.CDFT.get_number_of_electrons_on_atoms` helps in this
process.

.. autoclass:: gpaw.cdft.cdft.CDFT
    :members:


Optimizing `V_c`
----------------

Updating and optimizing the Lagrange multipliers `V_c` is achieved using
Scipy optimization routines. As it is easy to compute the gradient of the
energy wrt `V_c`, gradient based optimizers are recommended. The best
performing optimizer seems to be the :literal:`L-BFGS-B`. The accuracy of the
optimization is controlled mainly by the
:literal:`minimizer_options={'gtol':0.01})` parameter which measures the
error between the set and computed charge/spin value. A typical
:literal:`gtol` value is around 0.01-0.1.


Choosing the constraint values and regions
------------------------------------------

Both charge and spin constraints can be specified. The charge constraints are
specified to neutral atoms: specifying :literal:`charges = [1]` constrains
the first regions to have a net charge of +1. Note, that if the usual DFT
calculation yields e.g. a Fe ion with charge +2.5, specifying the charge
constraint +1 will result in +1, not an additional hole on Fe with a charge
state +3.5!

A constrained regions may contain several atoms and also several constrained
regions can be specified. However, converging more than two regions is
usually very difficult.


Tips for converging cDFT calculations
-------------------------------------

Unfortunately, the cDFT sometimes exhibits poor convergence.

1. Choose a meaningful constraints and regions

2. Try to provide a good initial guess for the `V_c`. In the actual
   calculation this initial guess given by the parameter
   :literal:`charge_coefs` or :literal:`spin_coefs`.

3. Use L-BFGS-B to set bounds for `V_c` by using
   :literal:`minimizer_options={'bounds':[min,max]})`.

4. Converge the underlying DFT calculation well i.e. use tight convergence
   criteria.


Constructing diabatic Hamiltonians
==================================

One of the main uses for cDFT is constructing diabatic Hamiltonians which
utilize cDFT states as the diabats. For instance, a 2x2 Hamiltonian matrix
would have diagonal elements `H_{11}` and `H_{22}` as well as off-diagonals
`H_{12}` and `H_{21}`. `H_{11}` and `H_{22}` values are given directly by
cDFT energies (not cDFT free energies). The off-diagonal coupling elements
`H_{12}` and `H_{21}` are also needed and often utilized in e.g.
Configuration Interaction-cDFT, in parametrizing model Hamiltonians or in
computing non-adiabatic charge transfer rates. Note, that all parameters need
to be computed at the same geometry.

The coupling elements are computed using the CouplingParameters class. There
are several options and optional inputs. There are two main methods for
computing coupling elements: the original cDFT approach from [#cdft1]_ and a
more general overlap or Migliore method detailed in [#cdft3]_.


cDFT coupling constant
----------------------

This method is the original approach in the context of cDFT. It works well
for simple system. For complex system it becomes quite sensitive to small
errors in the Lagrange multipliers and constraints, and the overlap method is
recommended instead. The inputs for the calculation are two cDFT objects with
different constraints. The coupling constant is computed using
:literal:`get_coupling_term`.


Overlap coupling constant
-------------------------

This approach has been found to perform very well for a range of systems.
However, it has one major limitation: it can only be used if the two diabatic
states have different energies. In addition to the two cDFT states/objects
also a normal DFT calculator without any constraints is needed. The coupling
constant is computed using :literal:`get_migliore_coupling`.


Additional comments
-------------------

In [#cdft2]_ the coupling constants were computed by computing all-electron
wave functions on a grid. However, this is quite slow and much faster
implementation utilizing only pseudo wave functions and atomic corrections
has been added. For the tested cases both give equally good values for the
coupling constant. Hence, it is recommended to use the pseudo wave functions
which is set by :literal:`AE=False`.

The quantities needed for computing the coupling constants can be parallelized
only over the grid.


Example of hole transfer in `He_2^+`
====================================

Both the cDFT calculation and extraction of the coupling element calculation
are demonstrated for the simple `He_2^+` system.

.. literalinclude:: He2.py

The most important cDFT results are found in the .cdft files. For instance,
the errors, iterations, and cDFT parameters are shown. Also, the energy can
be found in this file. The most relevant energy is the Final cDFT Energy (not
the free energy).


References
==========

.. [#cdft1] B. Kaduk, T. Kowalczyk, T. Van Voorhis,
            :doi:`Constrained Density Functional Theory <10.1021/cr200148b>`,
            *Chem. Rev.*, **112** 321–370 (2012)

.. [#cdft2] M. Melander, E. Jońsson, J.J. Mortensen, T. Vegge,J.M. Garcia-Lastra,
            :doi:`Implementation of Constrained DFT for Computing Charge Transfer Rates within the Projector Augmented Wave Method <10.1021/acs.jctc.6b00815>`,
            *J. Chem. Theory Comput.*, **12**, 5367−5378 (2016)

.. [#cdft3] A. Migliore,
        :doi:`Nonorthogonality Problem and Effective Electronic Coupling Calculation: Application to Charge Transfer in π-Stacks Relevant to Biochemistry and Molecular Electronics <10.1021/ct200192d>`,
        *J. Chem. Theory Comput.*, **7**, 1712-1725 (2011)
