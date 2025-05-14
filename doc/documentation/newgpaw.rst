.. _newgpaw:

“New GPAW”
==========

The GPAW backend is currently undergoing significant refactoring.
Occasionally we distinguish between “new” and “old” GPAW
in the documentation.

.. seealso::

   `Modernizing the GPAW code
   <https://jensj.gitlab.io/talks/dev24/talk.html>`__

To create a GPAW calculator using the new backend, use::

  from gpaw.new.ase_interface import GPAW as NewGPAW

To explicitly use the old backend, use::

  from gpaw.calculator import GPAW as OldGPAW

Default is to use old GPAW unless the environment variable
:envvar:`GPAW_NEW` is set.

.. _newparameters:

---------------------------
Parameters only in new GPAW
---------------------------

For a full list of paramters in old GPAW see :ref:`parameters`.

.. list-table::
    :header-rows: 1
    :widths: 1 1 1 2

    * - keyword
      - type
      - default value
      - description
    * - ``d3``
      - ``Work in progress``
      - ``None``
      - :ref:`d3correction`

End of table