.. _elphraman:

=======================================
Raman spectroscopy for extended systems
=======================================

This tutorial shows how to calculate the Raman spectrum of bulk MoS_2
(:download:`MoS2_2H_relaxed_PBE.json`)
using the electron-phonon coupling based Raman methods as described in
:ref:`raman`.


Effective Potential
===================

At the heart of the electron-phonon coupling is the calculation of the gradient
of the effective potential, which is done using finite displacements just like the
phonons. Those two calculations can run simultaneous, of the required set of
parameters coincide.

The electron-phonon matrix is quite sensitive to self-interaction of a displaced
atom with its periodic images, so a sufficiently large supercell needs to be used.
The ``(3x3x2)`` supercell used in this example should be barely enough. When using a supercell for the calculation you have to consider, that the atoms object needs
to contain the primitive cell, not the supercell, while the parameters for the
calculator object need to be good for the supercell, not the primitive cell.
(:git:`~doc/tutorialsexercises/vibrational/elphraman/displacement.py`)

.. literalinclude:: displacement.py

This calculation merely dumped the effective potential at various displacements
onto the harddrive. We now need to calculate the actual derivative and project them onto a set of LCAO basis functions.

For this we first need to complete a ground-state calculation for the supercell. This calculation needs to be done in LCAO mode with
parallelization over domains and bands disabled. (:git:`~doc/tutorialsexercises/vibrational/elphraman/supercell.py`)

.. literalinclude:: supercell.py

The ``calculate_supercell_matrix()`` method will then compute the gradients and
calculate the matrix elements. The results are saved in a file cache in a
basis of LCAO orbitals and supercell indices.


If you use the planewave mode for the displacement calculation, please see the note in :ref:`elph`.

Phonons
=======

The phonon frequencies enter the Raman calculation by defining the line positions in the final spectrum. Their associated mode vectors are necessary to project the
electron-phonon matrix from Cartesian coordinates in the modes.

It is crucial to use tightly relaxed structures and well-converged parameters to
obtain accurate phonon frequencies. We already calculated the forces in the previous step and need only the extract the phonon frequencies for later usage.
(:git:`~doc/tutorialsexercises/vibrational/elphraman/phonons.py`)

.. literalinclude:: phonons.py
    :start-at: # Phonon calculation
    :end-at: np.save

As exercise, check the dependence of the phonon frequencies with the calculation
mode, supercell size and convergence parameters.

 This given set of parameters yield this result::

    i    cm^-1
  ------------
    0   -11.21
    1    -2.75
    2     3.20
    3    29.01
    4    29.57
    5    55.19
    6   276.02
    7   276.44
    8   277.94
    9   278.39
   10   370.99
   11   371.43
   12   371.69
   13   371.95
   14   399.12
   15   402.67
   16   456.72
   17   460.57


Momentum matrix
===============

The Raman tensor does not only depend on the electron-phonon matrix, but also
on the transition probability between electronic states, which in this case is
expressed in the momentum matrix elements.
Those can be extracted directly from a converged LCAO calculation (:git:`~doc/tutorialsexercises/vibrational/elphraman/scf.py`) of the primitive
cell. (:git:`~doc/tutorialsexercises/vibrational/elphraman/dipolemoment.py`)

.. literalinclude:: dipolemoment.py


Phonon mode projected electron-phonon matrix
============================================

With all the above calculations finished we can extract the electron-phonon
matrix in the Bloch basis of the primitive cell projected onto the phonon modes:
(:git:`~doc/tutorialsexercises/vibrational/elphraman/gmatrix.py`)

.. literalinclude:: gmatrix.py

This will save the electron-phonon matrix as a numpy file to the disk.
The optional ``load_sc_as_needed`` tag prevents from all supercell cache files
being read at once. Instead they are loaded and processed one-by-one. This can
save lots of memory for larger systems with hundreds of atoms, where the
supercell matrix can be over 100GiB large.


Note: This part has not been tested properly for parallel runs and should be done
in serial mode only.


Raman spectrum
==============

With all ingredients provided we can now commence with the computation of the
Raman tensor which is saved in a file cache.(:git:`~doc/tutorialsexercises/vibrational/elphraman/raman.py`)

.. literalinclude:: raman.py

The final result can then be plotted:(:git:`~doc/tutorialsexercises/vibrational/elphraman/plot_spectrum.py`)

.. literalinclude:: plot_spectrum.py
    :start-at: from gpaw.elph import RamanData
    :end-before: # for testing

.. image:: Polarised_raman_488nm.png
    :scale: 30
    :align: center
