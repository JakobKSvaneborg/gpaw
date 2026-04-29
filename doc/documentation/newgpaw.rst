.. _newgpaw:

“New GPAW”
==========

The GPAW backend is currently undergoing significant refactoring.
Occasionally we distinguish between “new” and “old” GPAW
in the documentation.

.. note::

   New GPAW is now the default!

.. seealso::

   `Modernizing the GPAW code
   <https://jensj.gitlab.io/talks/dev24/talk.html>`__

For some operations, you will need to use old GPAW.
See the :ref:`release notes <new gpaw notes>` for which things this is.
To explicitly use the old backend, use::

  calc = GPAW(..., legacy_gpaw=True)


.. _extensions:

----------
Extensions
----------

.. toctree::
    :maxdepth: 1

    extensions/d3
