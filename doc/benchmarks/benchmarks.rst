==========
Benchmarks
==========

See :git:`gpaw/benchmark/performance_index.py`.

.. contents::


Test systems
============

.. csv-table::
    :file: systems.csv
    :header-rows: 1


Paramaters
==========

* PBE
* 800 eV plane-wave cutoff
* k-point density: 5.0 Å
* 14 electron potential for Cr
* default parameters for everything else


PW-mode performance index
=========================

The total time for one material is the sum of two steps
(`t=\Delta t_1 + \Delta t_2`):

1) time for for a complete SCF calculation
2) time for second SCF calculation after a small displacement
   of positions (or cell)

.. autofunction:: gpaw.benchmark.performance_index.score

.. image:: score.png


Results
=======

Figure shows:

* `t_i^0 / t_i`
* `\Delta t_2 / t)`
* Memory usage per core

.. image:: benchmark.png

Results for latest development version :

.. csv-table::
    :file: benchmark.csv
    :header-rows: 1


History
=======

2025, July
----------

* Initial run with 13 systems (score set to 100.0).
* Niflheim (``xeon24el8``, ``xeon40el8_clx``, ``xeon56``).
* Easy-build foss-2025b toolchain
  (Python-3.13.5, Numpy-2.3.2, Scipy-1.16.1, Libxc-7.0).


2025, November
--------------

* Added three more systems (``MnVS2-2M``, ``PtLi2O6-2M``, ``V3Cl6-2N``).
* Switched to :ref:`newgpaw`.


The future
----------

* Added three more systems (``ErGe-2M``, ``Mn2O2-3M``, ``Fe8O8-3M``).
