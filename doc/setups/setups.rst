.. _setups:
.. _datasets:

=================
Atomic PAW Setups
=================

A setup is to the PAW method what a pseudo-potential is to the
pseudo-potential method.  All available setups are contained in
:ref:`tar-files <setup releases>`.  There are setups for the LDA,
PBE, revPBE, RPBE and GLLBSC functionals.  Install them as described
in the :ref:`installation of paw datasets` section.  The setups are stored as
compressed :ref:`pawxml` files.


.. _setup releases:

Setup releases
==============

.. list-table::
   :header-rows: 1

   * - Date
     - Tarfile
     -
   * - Nov 27 2024
     - 24.11.0_
     - PAW potentials for lanthanides added.
       They have been generated with the following command::

         $ gpaw dataset La -sw -r2.2 -P5s,6s,5p,6p,5d,d,4f,f,G -fPBE -b

       and similarly for Ce, Pr, Nd, Pm, Sm, Eu, Gd,
       Tb, Dy, Ho, Er, Tm, Yb and Lu.
       There are also LDA versions of the potentials.

   * - Feb 22 2024
     - 24.1.0_
     - New 14 electron Cr PAW potential added.
       For high accuracy, it is recommended over the old 6-electron version
       (which is still the default).  You can use it by
       specifying ``setups={'Cr': '14'}`` (see also :ref:`manual_setups`).
       It has been generated with the following command::

         $ gpaw dataset Cr -sw -r2.0 -P3s,4s,3p,4p,3d,d,F -fPBE -t 14 -b

       There is also an LDA version of the potential.

   * - Mar 22 2016
     - 0.9.20000_
     -
   * - Mar 27 2014
     - 0.9.11271_
     -
   * - Oct 26 2012
     - 0.9.9672_
     -
   * - Apr 13 2011
     - 0.8.7929_
     -
   * - Apr 19 2010
     - 0.6.6300_
     -
   * - Jul 22 2009
     - 0.5.3574_
     -


.. _periodic table:

Periodic table
==============

=== === === === === === === === === === === === === === === === === ===
H_                                                                  He_
Li_ Be_                                         B_  C_  N_  O_  F_  Ne_
Na_ Mg_                                         Al_ Si_ P_  S_  Cl_ Ar_
K_  Ca_ Sc_ Ti_ V_  Cr_ Mn_ Fe_ Co_ Ni_ Cu_ Zn_ Ga_ Ge_ As_ Se_ Br_ Kr_
Rb_ Sr_ Y_  Zr_ Nb_ Mo_ Tc  Ru_ Rh_ Pd_ Ag_ Cd_ In_ Sn_ Sb_ Te_ I_  Xe_
Cs_ Ba_     Hf_ Ta_ W_  Re_ Os_ Ir_ Pt_ Au_ Hg_ Tl_ Pb_ Bi_ Po  At  Rn_
\
\
        La_ Ce_ Pr_ Nd_ Pm_ Sm_ Eu_ Gd_ Tb_ Dy_ Ho_ Er_ Tm_ Yb_ Lu_
=== === === === === === === === === === === === === === === === === ===

.. toctree::
   :glob:
   :hidden:

   [A-Z]*

:xkcd:`Old PAW-potentials <2975>` (no longer in use).


.. _installation of paw datasets:

Installation of PAW datasets
============================

A basic PAW dataset has been installed as part of the default
installation;
additional PAW datasets can be installed automatically or manually.

To install them automatically, run :command:`gpaw install-data
--{<dataset>} {<dir>}`.  This downloads and unpacks the newest package
into :file:`{<dir>}/{<name>}-{<version>}`.  When prompted, answer
yes (y) to register the path in the GPAW configuration file.

To manually install the setups, do as follows:

1) Get the tar file :file:`gpaw-setups-{<version>}.tar.gz`
   of the <version> of PAW datasets from the :ref:`setups` page
   and unpack it somewhere, preferably in ``$HOME``
   (``cd; tar -xf gpaw-setups-<version>.tar.gz``) - it could
   also be somewhere global where
   many users can access it like in :file:`/usr/share/gpaw-setups/`.
   There will now be a subdirectory :file:`gpaw-setups-{<version>}/`
   containing all the atomic data for the most commonly used functionals.

2) Set the environment variable :envvar:`GPAW_SETUP_PATH`
   to point to the directory
   :file:`gpaw-setups-{<version>}/`, e.g. for bash users, you would put into
   :file:`~/.bashrc`::

       export GPAW_SETUP_PATH=~/gpaw-setups-<version>

   Refer to :ref:`using_your_own_setups` for alternative way of
   setting the location of PAW datasets.

   .. note::

     In case of several locations of PAW datasets the first found setup
     file is used.

.. envvar:: GPAW_SETUP_PATH

    Colon-separated paths to folders containing the PAW datasets.

See also `NIST Atomic Reference Data`_.

.. _NIST Atomic Reference Data: https://physics.nist.gov/PhysRefData/DFTdata/Tables/ptable.html


Advanced topics
===============

.. toctree::
   :titlesonly:

   generation_of_setups
   pawxml


.. _acwf benchmark:

ACWF-benchmark
==============

Equation-of-state calculations for the 10 reference systems from
the `AiiDA common workflows (ACWF) benchmark <ACWF>`_:
DIAMOND, FCC, SC, BCC, XO3, XO, X4O6, XO2, X4O10, X2O.

See :git:`gpaw/utilities/acwf.py` for how to run these calculation.

The following table shows the errors in lattice constant in %
compared to accurate Wien2k results.  Calculations are done for:

* PW-mode calculations with ``ecut=1000`` (columns 3-5)
* LCAO-mode calculations with ``h=0.12`` (columns 6-8)

The coulumns are:

1. atomic number
2. PAW-potential name
3. number of errors (out of the 10 systems) (PW)
4. maxium absolute error (PW)
5. mean absolute error (PW)
6. number of errors (out of the 10 systems) (LCAO)
7. maxium absolute error (LCAO)
8. mean absolute error (LCAO)

.. csv-table::
   :file: acwf.csv
   :header: number, name, errors, max, mean, errors, max, mean

.. _ACWF: https://acwf-verification.materialscloud.org/


.. _24.11.0:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-24.11.0.tar.gz
.. _24.1.0:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-24.1.0.tar.gz
.. _0.9.20000:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.20000.tar.gz
.. _0.9.11271:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.11271.tar.gz
.. _0.9.9672:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.9672.tar.gz
.. _0.8.7929:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.8.7929.tar.gz
.. _0.6.6300:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.6.6300.tar.gz
.. _0.5.3574:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.5.3574.tar.gz

..  _H:  H.html
.. _He: He.html
.. _Li: Li.html
.. _Be: Be.html
..  _B:  B.html
..  _C:  C.html
..  _N:  N.html
..  _O:  O.html
..  _F:  F.html
.. _Ne: Ne.html
.. _Na: Na.html
.. _Mg: Mg.html
.. _Al: Al.html
.. _Si: Si.html
..  _P:  P.html
..  _S:  S.html
.. _Cl: Cl.html
.. _Ar: Ar.html
..  _K:  K.html
.. _Ca: Ca.html
.. _Sc: Sc.html
.. _Ti: Ti.html
..  _V:  V.html
.. _Cr: Cr.html
.. _Mn: Mn.html
.. _Fe: Fe.html
.. _Co: Co.html
.. _Ni: Ni.html
.. _Cu: Cu.html
.. _Zn: Zn.html
.. _Ga: Ga.html
.. _Ge: Ge.html
.. _As: As.html
.. _Se: Se.html
.. _Br: Br.html
.. _Kr: Kr.html
.. _Rb: Rb.html
.. _Sr: Sr.html
..  _Y: Y.html
.. _Zr: Zr.html
.. _Nb: Nb.html
.. _Mo: Mo.html
.. _Ru: Ru.html
.. _Rh: Rh.html
.. _Pd: Pd.html
.. _Ag: Ag.html
.. _Cd: Cd.html
.. _In: In.html
.. _Sn: Sn.html
.. _Sb: Sb.html
.. _Te: Te.html
..  _I:  I.html
.. _Xe: Xe.html
.. _Cs: Cs.html
.. _Ba: Ba.html
.. _La: La.html
.. _Ce: Ce.html
.. _Pr: Pr.html
.. _Nd: Nd.html
.. _Pm: Pm.html
.. _Sm: Sm.html
.. _Eu: Eu.html
.. _Gd: Gd.html
.. _Tb: Tb.html
.. _Dy: Dy.html
.. _Ho: Ho.html
.. _Er: Er.html
.. _Tm: Tm.html
.. _Yb: Yb.html
.. _Lu: Lu.html
.. _Hf: Hf.html
.. _Ta: Ta.html
..  _W:  W.html
.. _Re: Re.html
.. _Os: Os.html
.. _Ir: Ir.html
.. _Pt: Pt.html
.. _Au: Au.html
.. _Hg: Hg.html
.. _Tl: Tl.html
.. _Pb: Pb.html
.. _Bi: Bi.html
.. _Rn: Rn.html
