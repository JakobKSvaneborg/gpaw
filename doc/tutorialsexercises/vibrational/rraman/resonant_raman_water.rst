.. index:: resonant_raman_water
.. _resonant_raman_water:

===========================================
(Resonant-)Raman spectra of gas-phase water
===========================================

This tutorial shows how to calculate (resonant-)Raman spectra of a
single water molecule in the gas-phase. The theoretical background
can be found in Ref. [#WM20]_.


Accurate Forces from an IR calculation
======================================

A pre-condition for accurate forces and thus accurate vibrational frequencies
is relaxation with a rather small maximal force and a smaller gid-spacing
than is needed for excitated state calculations.
This is possible using the :class:`~ase.vibrations.Vibrations` or
:class:`~ase.vibrations.Infrared`
modules.

.. literalinclude:: H2O_ir.py

We can get the resulting frequencies

.. literalinclude:: H2O_ir_summary.py

with the result:

.. literalinclude:: H2O_ir_summary.txt

Only the last three vibrations are meaningful.


Static Raman from polarizabilities
==================================

The photon energy of the excitation laser in Raman experiments is often
well below electronic excitations of the molecules. Then the 
approximation to obtain Raman spectra from static polarizabilities
is valid. Here, we use the static polarizability obtained from external electric fields.

.. literalinclude:: H2O_static_raman.py

We can get the resulting frequencies and intensities

.. literalinclude:: H2O_Placzek_static_summary.py

with the result:

.. literalinclude:: H2O_Placzek_static_summary.txt

Again, only the last three vibrations are meaningful.


Resonant Raman: Excitations at each displacement
================================================

We need to calculate the excitations at each displacement and use
linear response TDDFT for this. This is the most time consuming
part of the calculation an we therefore use the coarser grid spacing
of :literal:`h=0.25`. We restrict to the first excitations
of the water molecule by setting
:literal:`{'restrict': {'energy_range': erange, 'eps': 0.4}}`.
Note, that the number of bands in the calculation is connected
to this.

.. literalinclude:: H2O_rraman_calc.py


Raman intensities
-----------------

We have to choose an approximation to evaluate the Raman intensities.
The most common is the Placzek approximation which we also apply here.
We may use :literal:`summary()` similar to :class:`~ase.vibrations.Infrared`,
but the Raman intensity depends on the excitation frequency.

.. literalinclude:: H2O_rraman_summary.py

with the result:

.. literalinclude:: H2O_rraman_summary.txt

Note, that the absolute intensity [#WM20]_ is given in the summary.


Raman spectrum
--------------

.. image:: H2O_rraman_spectrum.png

The Raman spectrum be compared to experiment shown above
can be obtained with the following script

.. literalinclude:: H2O_rraman_spectrum.py

The figure shows the sensitivity of relative peak heights on the
scattered photons energy.


References
==========

.. [#WM20] M. Walter and M. Moseler,
           :doi:`Ab-initio wave-length dependent Raman spectra: Placzek approximation and beyond <10.1021/acs.jctc.9b00584>`,
           *J. Chem. Theory Comput.* **16** (2020) 576-586J
