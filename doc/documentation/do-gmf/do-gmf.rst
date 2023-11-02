.. _do-gmf:

==================================================================================
Excited State Calculations with Direct Optimization and Generalized Mode Following
==================================================================================
The direct optimization generalized mode following (DO-GMF) method can be used to
perform variational calculations of excited electronic states, where, contrary to
:ref:`linear response TDDFT <lrtddft>`, the orbitals are variationally optimized
for the excited state.

The main challenge of variational density functional calculations of excited states
is that excited states often correspond to saddle points on the surface describing
the variation of the energy as a function of the electronic degrees of freedom (the orbital
variations). :ref:`Standard self-consistent field (SCF) algorithms  <manual_eigensolver>`
typically perform well in ground state calculations, as the latter is a minimum of the energy,
but face convergence issues in excited state calculations. As an alternative,
direct optimization (DO) approaches can be used, which have been found to
converge more robustly than the standard eigensolvers for excited states, especially in the
vicinity of electronic degeneracies. One option is to use quasi-Newton algorithms that
can converge to saddle points of arbitrary order in conjunction with the
:ref:`maximum overlap method (MOM) <mom>`, which can reduce the risk of converging to a
minimum or lower-energy saddle point (variational collapse). This is the DO-MOM method
implemented in GPAW and illustrated :ref:`here <directopt>`. However, DO-MOM can still
be affected by variational collapse in challenging cases. GPAW also implements an
alternative DO approach using a generalized mode following (GMF) method. DO-GMF
targets a stationary solution with a specific saddle point order and is more robust
than both DO-MOM and the standard SCF algorithms, while being inherently free from
variational collapse. On the other hand, DO-GMF has a bigger computational cost
than DO-MOM, because it requires more energy/gradient evaluations per iteration
due to the partial diagonalization of the Hessian.

--------------------------
Generalized mode following
--------------------------

~~~~~~~~~~~~~~
Implementation
~~~~~~~~~~~~~~

The implementation of the DO-GMF method is presented in [#dogmfgpaw1]_. For the moment,
the method can be used only in the LCAO mode.

GMF is a generalization of the minimum mode following method traditionally used to
optimize first-order saddle points on the potential energy surface for atomic rearrangements.
The method recasts the challenging saddle point search as a minimization by
inverting the projection of the gradient on the lowest eigenmode of the Hessian. It is
generalized to target an `n`-th-order saddle point on the electronic energy surface by
inverting the projections on the eigenmodes, `v_i`, of the electronic Hessian corresponding
to the `n` lowest eigenvalues, yielding the modified gradient

.. math::
    g^{\mathrm{\,mod}} = g - 2\sum_{i = 1}^{n}v_{i}v_{i}^{\mathrm{T}}g

if the energy surface is concave along all target eigenvectors, or

.. math::
    g^{\mathrm{\,mod}} = -\sum_{i = 1, \lambda_{i} > 0}^{n}v_{i}v_{i}^{\mathrm{T}}g

if any target eigenvalue, `\lambda_i`, is positive. Notice that in the latter case
only the target eigenvectors along which the energy surface is convex are followed
to increase stability of the method. The target eigenvalues and eigenvectors of the
electronic Hessian matrix are obtained by using a finite difference generalized Davidson
method [#gendavidson]_. This method can also be used for stability analysis of an
electronic solution (see :ref:`stabanalysisexample` below).

~~~~~~~~~~~~~~~~~
How to use DO-GMF
~~~~~~~~~~~~~~~~~

To provide initial guess orbitals for an excited state DO-GMF calculation, a ground
state calculation is typically performed first. Then, a DO-GMF calculation can be
requested as follows::

  from gpaw.directmin.lcao_etdm import LCAOETDM

  calc.set(eigensolver=LCAOETDM(
           partial_diagonalizer={'name': 'Davidson', 'logfile': None},
           linesearch_algo={'name': 'max-step'},
           searchdir_algo={'name': 'l-bfgs-p_gmf'},
           need_init_orbs=False),
           occupations={'name': 'mom', 'numbers': f,
                        'use_fixed_occupations': True})

where a log file for the partial Hessian diagonalization can be specified and ``f`` contains
the occupation numbers of the excited state (see :ref:`ethyleneexample` and :ref:`tPPexample`).
Line search algorithms cannot be applied for saddle point searches, so a maximum step length is
used. Any of the search direction algorithms implemented in GPAW (see :ref:`directmin`) can be
used by appending ``_gmf`` to the ``name`` keyword of the ETDM search direction algorithms
(e.g. specify ``l-bfgs-p_gmf`` to use the ``l-bfgs-p`` search direction with GMF).

A helper function can be used to create the list of excited state occupation numbers::

  from gpaw.directmin.tools import excite
  f = excite(calc, i, a, spin=(si, sa))

which will promote an electron from occupied orbital ``i`` in spin
channel ``si`` to unoccupied orbital ``a`` in spin channel ``sa``
(the index of HOMO and LUMO is 0). For example,
``excite(calc, -1, 2, spin=(0, 1))`` will remove an electron from
the HOMO-1 in spin channel 0 and add an electron to LUMO+2 in spin
channel 1.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Estimating the saddle point order of the target excited state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DO-GMF method requires an estimation of the saddle point order of the sought excited state
ahead of the actual calculation. GPAW estimates the saddle point order at the initial guess using
the following efficient diagonal approximation of the electronic Hessian:

.. math::
   :label: eq:hessapprox

    \mathscr{H}_{ijij} \approx 2\left(f_{j} - f_{i}\right)\left(\epsilon_{i} - \epsilon_{j}\right)

where `f_{i}` and `\epsilon_{i}` are the orbital occupation numbers and energies of the
initial guess orbitals, respectively. This approximation gives one negative eigenvalue for each
pair of occupied-unoccupied orbitals where the unoccupied orbital has lower energy than the occupied
one. For example, for a calculation initialized from an excitation from the ground state HOMO to
the ground state LUMO + 1, there will be two unoccupied orbitals (ground state HOMO and LUMO)
lower in energy than an occupied orbital (ground state LUMO + 1) and therefore the estimated
saddle point order is 2. This is usually a good estimation for low-lying valence and Rydberg
excitations.

For excitations involving significant charge transfer (see :ref:`tPPexample`), the energy ordering
of the orbitals of the converged solution can differ from the order of the initial guess orbitals.
In such cases, the diagonal Hessian approximation at the initial guess does not provide a good enough
estimation of the saddle point order. As shown in :ref:`tPPexample`, a better estimation is given
by first performing a constrained optimization with DO-MOM and then evaluating the saddle point
order using either the diagonal approximation of the Hessian or partial diagonalization of the full
Hessian (the latter is preferred). This can be done using::

 from gpaw.directmin.derivatives import Davidson

 davidson = Davidson(calc.wfs.eigensolver, eps=1e-2, seed=42)
 appr_sp_order = davidson.estimate_sp_order(calc, method='full-hess', target_more=3)

The estimated saddle point order then needs to be specified when requesting a DO-GMF calculation::

  from gpaw.directmin.lcao_etdm import LCAOETDM

  calc.set(eigensolver=LCAOETDM(
           partial_diagonalizer={'name': 'Davidson',
                                 'sp_order': appr_sp_order},
           ...)

.. _ethyleneexample:

-------------------------------------------
Example I: Doubly excited state of ethylene
-------------------------------------------
In this example, the lowest doubly excited state of ethylene is obtained with the DO-GMF method.
First, a ground state calculation is performed and then the DO-GMF calculation is initialized by
promoting one electron from the HOMO to the LUMO in both spin channels simultaneously. According
to the diagonal Hessian approximation, eq. :any:`eq:hessapprox`, the excited state is targeted as
a second-order saddle point on the electronic energy surface.

.. literalinclude:: ethylene.py

It is recommended to deactivate updates of the reference orbitals by setting the
``update_ref_orbs_counter`` keyword to a large value (e.g. 1000). The unitary invariant
representation should be used (if the density functional is orbital density independent) because
the redundant rotations among the occupied orbitals introduce many degenerate eigenvectors of the
electronic Hessian with zero curvature, which can lead to convergence problems of the generalized
Davidson method. The keyword ``use_fixed_occupations`` is set to ``True`` to deactivate the use of
the maximum overlap method, which is not needed here because variational collapse is impossible
with the DO-GMF method.

.. _tPPexample:

------------------------------------------------------------
Example II: Charge transfer excited state of N-phenylpyrrole
------------------------------------------------------------
In this example, a charge transfer excited state of the N-phenylpyrrole molecule is
calculated using the DO-GMF method. Since the target state is open-shell, the calculation
gives the energy of a mixed-spin solution. The energy of the mixed-spin solution can be
purified as shown in :ref:`h2oexample`, but this is not done in this example.

The excited state calculation is initialized by a single electron excitation from the
HOMO to the LUMO in one spin channel using the ground state orbitals. This target saddle point
order cannot be estimated using eq. :any:`eq:hessapprox` because the charge transfer excitation
leads to a large energetic rearrangement of the orbitals. To take this energetic rearrangement
into account and achieve a better estimation of the saddle point order, we first perform a
constrained optimization with DO-MOM freezing the hole and excited electron and minimizing all
other electronic degrees of freedom (see also :ref:`directopt`). Then, the saddle point order
is estimated from partial diagonalization of the full Hessian.

.. literalinclude:: estimate_sp_order.py

The saddle point order estimated by partial diagonalization of the Hessian is 9. However, closer
inspection of the negative eigenvalues from the log file of the Davidson calculation reveals that two
of them are significantly closer to 0 than the others, pointing towards a target saddle point order closer
to 7 rather than 9. It is then recommended to perform three trial calculations targeting saddle points of order `n-1`,
`n` and `n+1`, respectively, where `n` is the estimated approximate saddle point order (here 7).
Finally, the wanted excited state solution needs to be identified by inspecting the character of each of
the calculated solutions. Below we target with DO-GMF a sixth-order saddle point only, corresponding to the
calculation with `n-1`, because the wanted solution has been previously identified to be a sixth-order saddle point.

.. literalinclude:: tPP.py

DO-GMF converges to a sixth-order saddle point with a dipole moment of -10.227 D consistent with
the charge transfer character of the wanted excited state. Note that an unconstrained optimization
of this excited state with DO-MOM starting form an initial guess made of ground state orbitals
leads to variational collapse to a lower-energy saddle point with pronounced mixing between the HOMO
and LUMO and a small dipole moment of -3.396 D [#dogmfgpaw1]_.

.. _stabanalysisexample:

-----------------------------------------------------------------------------------
Example III: Stability analysis and breaking instability of ground state dihydrogen
-----------------------------------------------------------------------------------
In this example, the generalized Davidson method is used for stability analysis of the
ground state of the dihydrogen molecule. The molecule is stretched beyond the
Coulson-Fischer point, at which both a ground state solution with conserved symmetry and
two lower-energy degenerate ground state solutions with broken spin symmetry exist. First,
a spin-polarized direct minimization is performed starting from the GPAW default initial guess
for the orbitals. Stability analysis confirms that the obtained solution is a first-order
saddle point on the electronic energy surface, meaning that the symmetry-conserving
solution is obtained. Second, the electronic structure is displaced along the eigenvector
of the electronic Hessian corresponding to its lowest, negative eigenvalue, and thereby,
the instability is broken. This displaced electronic structure is reoptimized yielding a
lower-energy solution with broken spin symmetry. Stability analysis is applied to this
solution to confirm that it is a minimum on the electronic energy surface.

.. literalinclude:: H2_instability.py

----------
References
----------

.. [#dogmfgpaw1] Y. L. A. Schmerwitz, G. Levi, H. Jónsson
               :doi:`Calculations of Excited Electronic States by Converging on Saddle Points Using Generalized Mode Following <10.48550/ARXIV.2302.05912>`, (2023).

.. [#gendavidson] M. Crouzeix, B. Philippe, M. Sadkane
               :doi:`The Davidson Method <10.1137/0915004>`, *SIAM J. Sci. Comput.*, (1994).
