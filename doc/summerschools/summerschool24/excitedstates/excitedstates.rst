.. _excited states:

==============
Excited States
==============


This exercise investigates the properties and usability of
several materials in terms of applications in photovoltaics.

The exercise will teach you how to set up your material
and investigate the most important parameters
like the band gap and the absorption spectrum from
first principles calculations.
The first part shows how to calculate a converged band structure.
In the second part you will learn to calculate the quasiparticle
band gap. The third part involves learning how to calculate
an absorption spectrum within the random phase approximation.
Finally, you will calculate the absorption spectrum including
excitonic effects and compare different materials in the
fourth part.


Part 1: Setup of the structure and bandstructure calculations
=============================================================

The notebook :download:`es1.ipynb` shows how to set up the
material and how to calculate a converged band structure.

* Set up the atomic structure and optimize its geometry

* Calculate the band gap, band gap position,
  and band structure

* Compare the performance of different exchange correlation
  functionals


Part 2: Quasiparticle bandgap
=============================

The notebook :download:`es2.ipynb` teaches how to set up
calculations to find the quasiparticle band gap using GW
approximation.

* Understanding the GW approximation.

* First basic tests on the convergence of quasiparticle
  spectrum.

* Write/submit batch jobs


Part 3: Absorption spectrum
===========================

The notebook :download:`es3.ipynb` teaches how to set up
calculations of the dielectric function to find the absorption spectrum.

* Understanding the dielectric function

* First basic tests on the convergence of absorption
  spectra in the random phase approximation

* Write/submit batch jobs


Part 4: Excitonic effects and Discussion
========================================

In the last notebook :download:`es4.ipynb` the results for different
materials are plotted and discussed. Next the absorption spectra
is calculated including the excitonic effects (Bethe-Salpeter formalism).

* Understanding the excitonic effects and Bethe-Salpeter formalism.

* Basic tests on convergence of the absorption spectra.

* Write/submit batch jobs

Special care is taken of the convergence.
