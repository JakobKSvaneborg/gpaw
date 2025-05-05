.. _do:

==============================================================================
Excited-State Calculations with Direct Optimization
==============================================================================

Direct Optimization (DO) can be used to perform
variational calculations of excited states. Since
MOM calculations are variational, atomic forces are readily
available from the method ``get_forces`` and can, therefore,
be used to perform geometry optimization and molecular dynamics
in the excited state.

Excited-state solutions of the SCF equations are obtained
for non-Aufbau orbital occupations. MOM is a simple strategy to
choose non-Aufbau occupation numbers consistent
with the initial guess for an excited state during
optimization of the wave function, thereby facilitating convergence
to the target excited state and avoiding variational collapse to
lower energy solutions.

Even if MOM is used, an excited-state calculation can still be difficult
to convergence with the SCF algorithms based on diagonalization of the Hamiltonian
matrix that are commonly employed in ground-state
calculations. One of the main problems is that excited states
often correspond to saddle points of the energy as a function of the electronic
degrees of freedom (the orbital variations), but these algorithms perform better
for minima (ground states usually correspond to minima).
Moreover, standard SCF algorithms tend to fail when degenerate or nearly
degenerate orbitals are unequally occupied, a situation that is
more common in excited-state rather than ground-state calculations
(see :ref:`coexample` below).
In GPAW, excited-state calculations can be performed via a direct
optimization of the orbital (implemented for the moment only
in LCAO). DO can converge to a generic stationary point,
and not only to a minimum and has been shown to be more robust than diagonalization-based
:ref:`SCF algorithms <manual_eigensolver>` using density mixing in excited-state
calculations of molecules [#momgpaw1]_ [#momgpaw2]_ [#momgpaw3]_;
therefore, it is the recommended method for obtaining excited-state solutions
with MOM.

Direct optimization (DO) can be performed using the implementation
of exponential transformation direct minimization (ETDM)
[#momgpaw1]_ [#momgpaw2]_ [#momgpaw3]_ described in :ref:`directmin`.
This method uses the exponential transformation and efficient quasi-Newton
algorithms to find stationary points of the energy in the space of unitary
matrices.

For excited-state calculations, the recommended quasi-Newton
algorithm is a limited-memory symmetric rank-one (L-SR1) method
[#momgpaw2]_ with unit step. In order to use this algorithm, the
following ``eigensolver`` has to be specified::

  from gpaw.directmin.lcao_etdm import LCAOETDM

  calc.set(eigensolver=LCAOETDM(searchdir_algo={'name': 'l-sr1p'},
                                linesearch_algo={'name': 'max-step',
                                                 'max_step': 0.20})

The maximum step length avoids taking too large steps at the
beginning of the wave function optimization. The default maximum step length
is 0.20, which has been found to provide an adequate balance between stability
and speed of convergence for calculations of excited states of molecules
[#momgpaw2]_. However, a different value might improve the convergence for
specific cases.

If the target excited state shows pronounced charge transfer character, variational
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


..  _coexample:


----------------------------------------------------------------
Geometry relaxation excited-state of carbon monoxide
----------------------------------------------------------------

In this example, the bond length of the carbon monoxide molecule
in the lowest singlet `\Pi(\sigma\rightarrow \pi^*)` excited state
is optimized using two types of calculations, each based on a
different approximation to the potential energy curve of an open-shell
excited singlet state.
The first is a spin-polarized calculation of the mixed-spin state
as defined in :ref:`h2oexample`. The second is a spin-paired calculation
where the occupation numbers of the open-shell orbitals are set
to 1 [#levi2018]_. Both calculations use LCAO basis and the
direct optimization (DO) method.

In order to obtain the correct angular momentum
of the excited state, the electron is excited into a complex
`\pi^*_{+1}` or `\pi^*_{-1}` orbital, where +1 or −1 is the
eigenvalue of the z-component angular momentum operator. The
use of complex orbitals provides an excited-state density
with the uniaxial symmetry consistent with the symmetry of the
molecule [#momgpaw1]_.

.. literalinclude:: domom_co.py

The electronic configuration of the `\Pi(\sigma\rightarrow \pi^*)`
state includes two unequally occupied, degenerate `\pi^*` orbitals.
Because of this, convergence to this excited state is more
difficult when using SCF eigensolvers with density mixing
instead of DO, unless symmetry constraints on the density
are enforced during the calculation. Convergence of such
excited-state calculations with an SCF eigensolver can be
improved by using a Gaussian smearing of the holes and excited
electrons [#levi2018]_.
Gaussian smearing is implemented in MOM and can be used
by specifying a ``width`` in eV for the Gaussian smearing
function::

  mom.prepare_mom_calculation(..., width=0.01, ...)

For difficult cases, the ``width`` can be increased at regular
intervals by specifying a ``width_increment=...``.
*Note*, however, that too extended smearing can lead to
discontinuities in the potentials and forces close to
crossings between electronic states [#momgpaw2]_, so
this feature should be used with caution and only
at geometries far from state crossings.

..  _ppexample:

--------------------------------------------------------------------------------------
Constrained optimization charge transfer excited state of N-phenylpyrrole
--------------------------------------------------------------------------------------

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
consistent with the wanted charge transfer state.

.. literalinclude:: constraints.py


-----------------------------------------------------
Excited state of silicon using plane wave
-----------------------------------------------------

In this example, the singlet excited state of a conventional silicon is calculated using plane waves approach.

.. literalinclude:: si_es.py


----------
References
----------

.. [#momgpaw1] A. V. Ivanov, G. Levi, H. Jónsson
               :doi:`Method for Calculating Excited Electronic States Using Density Functionals and Direct Orbital Optimization with Real Space Grid or Plane-Wave Basis Set <10.1021/acs.jctc.1c00157>`,
               *J. Chem. Theory Comput.*, (2021).

.. [#momgpaw2] G. Levi, A. V. Ivanov, H. Jónsson
               :doi:`Variational Density Functional Calculations of Excited States via Direct Optimization <10.1021/acs.jctc.0c00597>`,
               *J. Chem. Theory Comput.*, **16** 6968–6982 (2020).

.. [#momgpaw3] G. Levi, A. V. Ivanov, H. Jónsson
               :doi:`Variational Calculations of Excited States Via Direct Optimization of Orbitals in DFT <10.1039/D0FD00064G>`,
               *Faraday Discuss.*, **224** 448-466 (2020).

.. [#levi2018] G. Levi, M. Pápai, N. E. Henriksen, A. O. Dohn, K. B. Møller
               :doi:`Solution structure and ultrafast vibrational relaxation of the PtPOP complex revealed by ∆SCF-QM/MM Direct Dynamics simulations <10.1021/acs.jpcc.8b00301>`,
               *J. Phys. Chem. C*, **122** 7100-7119 (2018).
