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
diag            Set up H and S and find all or some eigenvectors/values
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
analyzing :ref:`point groups`          :mod:`gpaw.point_groups`
:ref:`hyperfine`                       :mod:`gpaw.hyperfine`
Calculation of dipole matrix elements  :mod:`gpaw.utilities.dipole`
PAW-dataset convergence                :mod:`gpaw.utilities.ekin`
:ref:`elph`                            ``gpaw.elph.gpts``
=====================================  ============================

Try::

    $ python3 -m <module> --help


.. _bash completion:

Bash completion
===============

You can enable bash completion like this::

    $ gpaw completion

This will append a line like this::

    complete -o default -C /path/to/gpaw/gpaw/cli/complete.py gpaw

to your ``~/.bashrc``.
