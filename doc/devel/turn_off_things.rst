How to turn off things (and make development and debugging simpler)
===================================================================

When developing new features or debugging it can be an advantage to simplify
the problem by turning things off.


No PAW corrections
------------------

* For hydrogen we have an "all-electron" potential (bare Coulomb potential).
  Use ``setups='ae'``.  See :ref:`ae hydrogen atom` and
  :git:`gpaw/ae.py`.

* For silicon we have the Appelbaum-Hamann local pseudo potential.
  Use ``setups='ah'``.  See :git:`gpaw/test/pseudopotential/test_ah.py`
  and :git:`gpaw/ah.py`.

* For other elements we have norm-conserving non-local pseudo-potentials:
  :ref:`manual_setups`.


No XC functional
----------------

Use ``xc={'name': 'null'}``.


No Coulomb interactions
-----------------------

Use ``poissonsolver=NoInteractionPoissonSolver()``
(and ``from gpaw.poisson import NoInteractionPoissonSolver``).
