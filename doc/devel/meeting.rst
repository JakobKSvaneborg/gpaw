.. _dev-meeting:

=====================================
October-2024 online developer meeting
=====================================

When:
  October 8 and 9 afternoons (Danish time)
Announcement:
  https://listserv.fysik.dtu.dk/pipermail/gpaw-users/2024-September/007312.html

.. note::

   Talks are approximately 15 + 10 minutes.
   Please try to focus your talks on the equations and algorithms you need
   to solve and the challenges implementing those in the code.


Program
=======

Tuesday
-------

:14:00:

  **Jens Jørgen Mortensen**

  *Modernizing the code-base*

  Discussion of changes taking place in the code at the moment and how
  that will affect all of us in the future.

:14:25:

  **Aleksei Ivanov**

  *UPAW method, wavefunction theories and quantum computing prospects*

  In this presentation, I will introduce the unitary projector
  augmented-wave method (UPAW) and its generalization on many-body wave
  functions.  Then I will discuss the benefits UPAW brings in the
  development of the electronic structure algorithms with focus on
  many-body theories and quantum computation.  I will show a few
  examples on how one can calculate the MP2 energy with GPAW and the
  interface with other electronic structure software such as PySCF to
  carry out more complicated wavefunction calculations using, for
  example, coupled cluster theories.  I will finish the talk with
  prospects on using GPAW for quantum embedding theories such as the
  density matrix embedding theory.

:14:50:

  **Marko Melander**

  *Constant inner potential (CIP) DFT method (extension of SJM)*

  CIP-DFT and its further development as well as working towards
  integrating Poisson-Boltzmann models in the main GPAW version.

:15:15:

  **Colin Baker**

  *Porting SJM into planewave mode*

:15:40:

  **Kyle Bystrom**

  *PAW Implementation of Nonlocal, Machine-Learned Density Functionals*

  Description: I will discuss an approach for efficiently evaluating
  nonlocal density functionals within the PAW formalism and the
  application of this approach to machine learning-based functionals.
  I will also mention work in progress to further optimize this
  algorithm and plans to extend it to van der Waals functionals and
  orbital-dependent features.

:16:05:

  **Pooria Dabbaghi**

  *Implementing an inverse Kohn-Sham scheme within the PAW formalism*

:16:30:

  **To be announced**

  *Improved basis sets for LCAO*


Wednesday
---------

:14:00:

  **Mikael Kuisma**

  *Running calculations on GPUs*

:14:25:

  **Tuomas Rossi**

  *Compiling for AMD and NVIDIA*

:14:50:

  **Ask Hjorth Larsen**

  *ASR-LIB*

:15:15:

  **Gianluca Levi**

  *Direct optimization methods for excited state calculations*

  The current status of the implementation, what is missing and what
  are the new features that I'd like to implement next.

:15:40:

  **Gianluca Levi**

  *Self-interaction correction*

:16:30:

  **Vladimír Zobač**

  *Ehrenfest molecular dynamics with LCAO basis*

:16:05:

  **To be announced**

  *RTTDDFT*
