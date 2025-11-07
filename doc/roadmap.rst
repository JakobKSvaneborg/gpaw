=======
Roadmap
=======

Finish new GPAW
---------------

Make :ref:`newgpaw` feature-complete and remove old code.


Switch to spglib (maybe)
------------------------

Our current symmetry-analysis code has problems (see :mr:`2933`).
We could use spglib_ instead:

.. code:: python

    from spglib import get_symmetry_dataset
    data = get_symmetry_dataset((cell_cv, relpos_ac, ids),
                                symprec=tolerance)


.. _spglib: https://spglib.readthedocs.io/en/stable/


Use mpi4py (maybe)
------------------

Why have our own interface to MPI when we can just use mpi4py_?
See :mr:`2904`.


.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
