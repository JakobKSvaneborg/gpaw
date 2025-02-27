.. program:: gpaw
.. highlight:: bash
.. index:: gpaw, command line interface, CLI

.. _cli:

======================
Command line interface
======================

GPAW has a command line tool called :program:`gpaw` with the following
sub-commands:

==============  ==============================================================
sub-command     description
==============  ==============================================================
help            Help for sub-command
run             Run calculation with GPAW
info            Show versions of GPAW and its dependencies
dos             Calculate (projected) density of states from gpw-file
gpw             Write summary of GPAW-restart file
completion      Add tab-completion for Bash
atom            Solve radial equation for an atom
python          Run GPAW's parallel Python interpreter
sbatch          Submit a GPAW Python script via sbatch
dataset         Calculate density of states from gpw-file
symmetry        Analyse symmetry (and show IBZ **k**-points)
install-data    Install additional PAW datasets, pseudopotential or basis sets
==============  ==============================================================

Example::

    $ gpaw info


Help
====

You can do::

    $ gpaw --help
    $ gpaw sub-command --help

to get help (or ``-h`` for short).


Other command-line tools
========================

There are also CLI tools for:

=====================================  ============================
description                            module
=====================================  ============================
analysing :ref:`point groups`          :mod:`gpaw.point_groups`
:ref:`hyperfine`                       :mod:`gpaw.hyperfine`
:ref:`fulldiag`                        :mod:`gpaw.fulldiag`
Calculation of dipole matrix elements  :mod:`gpaw.utilities.dipole`
PAW-dataset convergence                :mod:`gpaw.utilities.ekin`
:ref:`elph`                            ``gpaw.elph.gpts``
=====================================  ============================

Try::

    $ python3 -m <module> --help


.. module:: gpaw.fulldiag
.. _fulldiag:

Finding all or some unocupied states
------------------------------------

If you have a gpw-file containing the ground-state density for a plane-wave
calculation, then you can set up the full
`H_{\mathbf{G}\mathbf{G}'}(\mathbf{k})` and
`S_{\mathbf{G}\mathbf{G}'}(\mathbf{k})` matrices in your plane-wave basis and
use direct diagonalization to find all the eigenvalues and eigenstates in one
step.

Usage::

    $ python3 -m gpaw.fulldiag [options] <gpw-file>

Options:

-h, --help            Show this help message and exit
-n BANDS, --bands=BANDS
                      Number of bands to calculate.  Defaults to all.
-s SCALAPACK, --scalapack=SCALAPACK
                      Number of cores to use for ScaLapack.  Default is one.
-d, --dry-run         Just write out size of matrices.

Typpically, you will want to run this in parallel and distribute the matrices
using ScaLapack::

    $ gpaw -P 8 python -m gpaw.fulldiag abc.gpw --scalapack=8 ...


.. _bash completion:

Bash completion
===============

You can enable bash completion like this::

    $ gpaw completion

This will append a line like this::

    complete -o default -C /path/to/gpaw/gpaw/cli/complete.py gpaw

to your ``~/.bashrc``.
