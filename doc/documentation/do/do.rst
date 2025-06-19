.. _do:

==============================================================================
Excited-State Calculations with Direct Optimization
==============================================================================

Direct optimization (DO) is an alternative to the diagonalization-based
:ref:`eigensolvers <manual_eigensolver>`, which does not use density mixing.
Since it is designed to converge to saddle point on the electronic energy surface,
this approach can be used to perform
variational calculations of excited states. The atomic forces can be obtained
directly from the method ``get_forces`` and therefore,
the method can be used to perform geometry optimization and molecular dynamics
in the excited state.

The implementation of DO is based on the exponential transformation
(see also :ref:`directmin`) and uses efficient quasi-Newton algorithms
for saddle point convergence.
The real-space grid and plan-wave implementation is described in [#do1]_,
while the LCAO inplementation is described in [#do2]_ and [#do3]_.

The recommended quasi-Newton algorithm is a limited-memory symmetric rank-one (L-SR1) method
[#do2]_ with unit step. In order to use this algorithm, the
following ``eigensolver`` has to be specified::

  from gpaw.directmin.lcao_etdm import LCAOETDM

  calc.set(eigensolver=LCAOETDM(searchdir_algo={'name': 'l-sr1p'},
                                linesearch_algo={'name': 'max-step',
                                                 'max_step': 0.20})

The maximum step length avoids taking too large steps at the
beginning of the wave function optimization. The default maximum step length
is 0.20, which has been found to provide an adequate balance between stability
and speed of convergence for calculations of excited states of molecules
[#do2]_. However, a different value might improve the convergence for
specific cases.

To reduce the risk of variational collapse to lower energy solutions, DO is typically used together
with the :ref:`mom`. However, if the target excited state shows pronounced charge transfer character, variational
collapse can sometimes not be prevented even if DO and MOM are used in conjunction. In
such cases, it can be worthwhile to first perform a constrained optimization in which the
electron and hole orbitals involved in the target excitation are frozen, and a
minimization is done in the remaining subspace, before performing a full unconstrained
optimization. The constrained minimization takes care of a large part of the prominent
orbital relaxation effect in charge transfer excited states and thereby significantly
simplifies the subsequent saddle point search, preventing variational collapse.
Constrained optimization can be performed by using the ``constraints`` keyword::

  calc.set(eigensolver=LCAOETDM(constraints=[[[h11], [h12],..., [p11], [p12],...],
                                             [[h21], [h22],..., [p21], [p22],...],
                                              ...])

Each ``hij`` refers to the index of the ``j``-th hole in the ``i``-th K-point,
each ``pij`` to the index of the j-th excited electron in the ``i``-th K-point.
For example, if an excited state calculation is initialize by promoting an electron
from the ground state HOMO to the ground state LUMO, one needs to specify the indices
of the ground state HOMO (hole) and LUMO (excited electron) in the spin channel where
the excitation is performed.
All rotations involving these orbitals are frozen during the constrained optimization
resulting in these orbitals remaining unaltered after the optimization. It is
also possible to constrain selected orbital rotations without completely
freezing the involved orbitals by specifying lists of two orbital indices
instead of lists of single orbital indices. However, care has to be taken
in that case since constraining a single orbital rotation may not fully
prevent mixing between those two orbitals during the constrained
optimization.


..  _ppexample:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Charge transfer excited state of N-phenylpyrrole
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, a calculation of a charge transfer excited state of the N-phenylpyrrole
molecule is carried out. After a ground state calculation, a single excitation is performed
from the HOMO to the LUMO in one spin channel. No spin purification is used, meaning that
only the mixed-spin open-shell determinant is optimized. If an unconstrained optimization
is performed from this initial guess, the calculation collapses to a first-order saddle point
with pronounced mixing between the HOMO and LUMO and a small dipole moment of -3.396 D, which
is not consistent with the wanted charge transfer excited state. Variational collapse is avoided
here by performing first a constrained optimization freezing the hole and excited electron
of the initial guess. Then the new orbitals are used as the initial guess of an unconstrained
optimization, which converges to a higher-energy saddle point with a large dipole moment of -10.227 D
consistent with the target charge transfer state.

.. literalinclude:: constraints.py


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Excited state of silicon using plane wave
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, the singlet excited state of a conventional silicon is calculated using plane waves approach.

.. literalinclude:: si_es.py


----------
References
----------

.. [#do1] A. V. Ivanov, G. Levi, H. Jónsson
               :doi:`Method for Calculating Excited Electronic States Using Density Functionals and Direct Orbital Optimization with Real Space Grid or Plane-Wave Basis Set <10.1021/acs.jctc.1c00157>`,
               *J. Chem. Theory Comput.*, (2021).

.. [#do2] G. Levi, A. V. Ivanov, H. Jónsson
               :doi:`Variational Density Functional Calculations of Excited States via Direct Optimization <10.1021/acs.jctc.0c00597>`,
               *J. Chem. Theory Comput.*, **16** 6968–6982 (2020).

.. [#do3] G. Levi, A. V. Ivanov, H. Jónsson
               :doi:`Variational Calculations of Excited States Via Direct Optimization of Orbitals in DFT <10.1039/D0FD00064G>`,
               *Faraday Discuss.*, **224** 448-466 (2020).
