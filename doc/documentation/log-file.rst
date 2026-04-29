Reading GPAW's log-file
=======================

ASE-plugin
----------

ASE's :func:`ase.io.read` function will use
:func:`gpaw.ase_plugin.read_gpaw_log` and
:func:`gpaw.io.log_file_reader.parse` to read atoms from GPAW's
log-file.

.. automodule:: gpaw.ase_plugin
.. autofunction:: gpaw.ase_plugin.read_gpaw_log


Parsing
-------

.. autofunction:: gpaw.io.log_file_reader.parse
.. autofunction:: gpaw.io.log_file_reader.parse_file
