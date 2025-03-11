.. _custom_convergence:

===========================
Custom convergence criteria
===========================

Additional convergence keywords
-------------------------------

There are additional keywords that you can provide to the ``convergence`` dictionary beyond those in the :ref:`default dictionary <manual_convergence>`.
These include ``'forces'``, ``'work function'``, and ``'minimum iterations'``.
(See :ref:`builtin_criteria` for a list of all available criteria and their parameters.)
For example, to make sure that the work function changes by no more than 0.001 eV across the last three SCF iterations, you can do::

  from gpaw import GPAW

  convergence={'work function': 0.001}

  calc = GPAW(...,
              convergence=convergence)

In the example above, the default criteria (energy, eigenstates, and density) will still be present and enforced at their default values.
The default convergence criteria are always active, but you can effectively turn them off by setting any of them to :code:`np.inf`.


Changing criteria behavior
--------------------------

You can change things about how some convergence criteria work through an alternative syntax.
For example, the default syntax of :code:`convergence={'energy': 0.0005}` ensures that the last three values of the energy change by no more than 5 meV.
If you'd rather have it examine changes in the last *four* values of the energy, you can set your convergence dictionary to::

  from gpaw.convergence_criteria import Energy

  convergence = {'energy': Energy(tol=0.0005, n_old=4)}

(In fact, :code:`convergence={'energy': 0.0005}` is just a shortcut to :code:`convergence={'energy': Energy(0.0005)}`; the dictionary value :code:`0.0005` becomes the first positional argument to :code:`Energy`.)

.. _custom_convergence-forces:

Converging forces
-----------------

You can ensure that the forces are converged like::

  convergence = {'forces': 0.01}

This requires that the maximum change in the magnitude of the vector representing the difference in forces for each atom is less than 0.01 eV/ Angstrom, compared to the previous iteration.
Since calculating the atomic forces takes computational time and memory, by default this waits until all other convergence criteria are met before beginning to check the forces.
If you'd rather have it check the forces at every SCF iteration you can instead do::

  from gpaw.convergence_criteria import Forces

  convergence = {'forces': Forces(0.01, calc_last=False)}

You can also choose to converge forces relative to the current maximum force acting on all atoms in your system. This is particularly useful for example in the case of geometry optimizations far from local minima where large forces mean that strict SCF (and therefore forces) convergence is not necessary. For this one can do::

  # Converge forces to 10% of the highest force.
  convergence = {'forces': Forces(atol=np.inf, rtol=0.1)}

If both ``atol`` and ``rtol`` are supplied, then forces are converged to whichever is the stricter convergence for that SCF cycle::

  # During a geometry optimization, converge forces to 0.01 eV/Ang
  # between successive SCF iterations until forces are below
  # 0.1 eV/Ang, then 10% of maximum force in system.
  convergence = {'forces': Forces(atol=0.01, rtol=0.1)}
  

Example: fixed iterations
-------------------------

You can use this approach to tell the SCF cycle to run for a fixed number of
iterations. To do this, set all the default criteria to :code:`np.inf` to
turn them off, then use the :class:`~gpaw.convergence_criteria.MinIter` class
to set a minimum number of iterations. (Also be sure your :ref:`maxiter
<manual_convergence>` keyword is set higher than this value!) For example, to
run for exactly 10 iterations::

  convergence = {'energy': np.inf,
                 'eigenstates': np.inf,
                 'density': np.inf,
                 'minimum iterations': 10}

The :class:`~gpaw.convergence_criteria.MinIter` class can work in concert
with other convergence criteria as well; that is, it can act simply to define
a minimum number of iterations that must be run, even if all other criteria
have been met.


Writing your own criteria
-------------------------

You can write your own custom convergence criteria if you structure them like this::

  from gpaw.convergence_criteria import Criterion


  class MyCriterion(Criterion):
      name = 'my criterion'  # must be a unique name
      tablename = 'mycri'  # <=5 char, prints as header in the SCF table
      calc_last = False  # if True, waits until all other criteria are met
                         # before checking (for expensive criteria)

      def __init__(self, ...):
          ...  # your code here; note if you save all arguments directly
               # (as self.a, self.b, ...) then todict() and __repr__ methods
               # will work automatically.
          # The next line prints at the top of the log file.
          self.description = 'My custom criterion with tolerance ...'

      def __call__(self, context):
          ...  # your code here
               # 'context' is an object containing references to the current
               # state of the calculation, such as the hamiltonian and wave
               # functions
          converged = ...  # True or False if your criterion is met
          entry = ...  # a string with up to 5 characters to print in SCF table
          return converged, entry

      def reset(self):
          ...  # your code here to clear anything saved whenever
               # the SCF restarts



  calc = GPAW(...,
              convergence={'custom': [MyCriterion(0.01, 4)]}
             )


All user-written criteria must enter the dictionary through the special ``custom`` keyword, and you can include as many criteria as you like in the list.

.. note::
   If you have written your own criterion and you save your calculator instance (that is, :code:`calc.write('out.gpw')`), GPAW won't know how to load your custom criterion when it opens"out.gpw".
   You will need to add your custom criteria back manually.

.. note::
   If you are running multiple GPAW calculator instances simultaneously, make sure each calculator instance gets its own unique instance of your custom criterion.
   (You do not need to worry about this for any of the built-in criteria, as it makes an internal copy.)


.. _builtin_criteria:

Built-in criteria
-----------------

The built-in criteria, along with their shortcut names that you can use to access them in the :code:`convergence` dictionary, are below.
The criteria marked as defaults are present in the default convergence dictionary and will always be present; the others are optional.

.. list-table::
    :header-rows: 1
    :widths: 1 1 1 1 1

    * - class
      - name attribute
      - default?
      - calc_last?
      - override_others?
    * - :class:`~gpaw.convergence_criteria.Energy`
      - ``energy``
      - Yes
      - No
      - No
    * - :class:`~gpaw.convergence_criteria.Density`
      - ``density``
      - Yes
      - No
      - No
    * - :class:`~gpaw.convergence_criteria.Eigenstates`
      - ``eigenstates``
      - Yes
      - No
      - No
    * - :class:`~gpaw.convergence_criteria.Forces`
      - ``forces``
      - No
      - Yes
      - No
    * - :class:`~gpaw.convergence_criteria.WorkFunction`
      - ``work function``
      - No
      - No
      - No
    * - :class:`~gpaw.convergence_criteria.MinIter`
      - ``minimum iterations``
      - No
      - No
      - No
    * - :class:`~gpaw.convergence_criteria.MaxIter`
      - ``maximum iterations``
      - No
      - No
      - Yes


Full descriptions for the built-in criteria follow.

.. autoclass:: gpaw.convergence_criteria.Energy
.. autoclass:: gpaw.convergence_criteria.Density
.. autoclass:: gpaw.convergence_criteria.Eigenstates
.. autoclass:: gpaw.convergence_criteria.Forces
.. autoclass:: gpaw.convergence_criteria.WorkFunction
.. autoclass:: gpaw.convergence_criteria.MinIter
.. autoclass:: gpaw.convergence_criteria.MaxIter
