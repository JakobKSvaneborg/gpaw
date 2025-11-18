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

Default parameters except:

* PBE
* 800 eV plane-wave cutoff
* k-point density: 5.0 Å
* 14 electron potential for Cr


PW-mode performance index
=========================

.. autofunction:: gpaw.benchmark.performance_index.score

.. image:: score.png


Results
=======

Figure shows:

* Time for for SCF calculation (`t_1`) plus time for second SCF
  calculation after small displacement of positions or cell (`t_2`)
* `t_2 / (t_1 + t_2)`
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
