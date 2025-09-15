.. _esstutorial:

=================================================================
Tutorial for ESS workshop 2025 - Magnetism and exchange constants
=================================================================


Magnetic moment and band structure of bcc Fe
============================================

We start by doing a few simple calculations of the magnetic
structure of bcc Fe. The following script performs the ground
state calculation

.. literalinclude:: gs_fe.py

It may also be downloaded here :download:`gs_fe.py`.
First, an ``Atoms`` object containing one Fe atom in a bcc
unit cell, is created. We initialize a magnetic moment of 2 Bohr
magnetons on the Fe atom and use periodic boundary conditions
(``pbc=True`` ). Next, a GPAW calculator with plane waves
and k-point sampling is constructed and attached to the ``Atoms``
object (bcc iron). The calculation is initialized with the call
to ``get_potential_energy()`` and the calculator will know that
it should do a spin-polarized calculation because of the initial
magnetic moment. The magnetization density and total magnetic
moment are, however, outputs from the calculations and will be
determined self-consistently.

The script can be run with the command::

    mpiexec -n 1 gpaw python gs_fe.py

The ``-n`` option refers to the number of CPU cores used in the
calculation. A single core will do for this one, which is rather
fast. Run the calculation and inspect the output.

 * How many iterations was required to converge the calculation?
   And how long did it take?

 * What is the total magnetic moment and the local magnetic moment
   resulting from the calculation (these are printed immediately
   after the iterative self-consistency cycle)? What is the
   difference between the two? Do they agree with the experimental
   magnetization of bulk iron?

The calculation finished by saving the results to a ``.gpw`` file.
This contains all the parameters used in the calculation as well as
the converged electronic density, Kohn-Sham eigenvalues and total
energy.

Next, we will calculate the band structure of bcc iron. This can be
done with the script :download:`bandstructure_fe.py`. Inspect the
script and note that it restarts from the previous calculation and
performs a non-self-consistent calculation using the
``fixed_density()`` method. Instead of a uniform *k*-point grid we
are now specifying a particular path (GHNG) in the Brillouin zone.
The script will plot the band structure and save the results
in a ``.gpw`` file. Run the script (a single core will suffice).

 * Can you identify the *s*-band and the *d*-band from visual
   inspection of the band structure?

 * What is (roughly) the exchange splitting of the *d*-band?

 * Would the calculation have been more accurate if we performed
   a self-consistent calculation on the given path in *k*-space?


Magnon dispersion and exchange constants of altermagnetic MnF2
==============================================================

The next material to consider is the anti-ferromagnet MnF2. We will
refer to it is a compensated magnet, since it has recently been reclassified
as being altermagnetic. We will not go into *that* discussion but will
calculate the magnon dispersion and show that certain (lack of) symmetries
renders the magnon dispersion non-degenerate in most of the Brillouin zone.
To start with we perform a ground state calculation in the spin-compensated
state. This can be done with the script:

.. literalinclude:: gs_mnf2.py

It may be downloaded here :download:`gs_mnf2.py` Similar to the case of bcc
iron we define a list of atoms and attach a
calculator (GPAW), which can perform the DFT calculation. However, instead
of setting up the structure (list of atoms) explicitly, we use the method
``crystal()``, which will generate the structure from the space group,
certain cell parameters and the Wyckoff positions. The
result is a list of atoms object with six atoms and we initialize
the magnetic moments in a spin compensated state. We also include a Hubbard
correction (LDA+U) on the Mn *d*-orbitals with a U of 6 eV
(``setups={'Mn': ':d,6.0'}``). Run the calculation. It will take about 1-2
minutes on six cores, Inspect the output written to ``gs_afm.txt``. If you
would like to visualize the structure you may do ``ase gui gs_afm.gpw`` or
``ase gui gs_afm.gpw``. The unit cell may be rotated in the gui for better
view or it may be repeated in different directions.

 * What are the local moments on the Mn atoms? Does it agree with
   expectations? What is the (Kohn-Sham) band gap?

Now modify the script ``gs_mnf2.py`` such that the system is initialized
with ferromagnetic alignment of magnetic moments. Run the calculation and
make sure to rename the output files such that the ones that were already
calculated are not overwritten.

 * What is the total magnetic moment? How does the local magnetic moments
   compare to the anti-ferromagnetic calculation? The local and total
   moments should convince you that it is reasonable to model this as a
   pure `S=5/2` magnetic lattice (disregarding the F atoms).

 * What is the energy difference per Mn atom compared to the
   anti-ferromagnetic calculations performed previously?

 * Let us assume that the system can be modelled by an isotropic Heisenberg
   model of the form

   .. math:: -\frac{1}{2}\sum_{abij}J_{ij}^{ab}
	     \mathbf{S}_i^a\cdot\mathbf{S}_j^b

   Here `\mathbf{S}_i^a` is the spin operator of magnetic atom `a` in unit
   cell `i` and `J_{ij}^{ab}` is the exchange coupling connecting the spin of
   `ia` to `jb`. Assume that the only exchange interaction is the closest
   inter-sublattice one, (between the two atoms in the unit cell), and use
   the DFT energy difference calculated above to provide an estimate of the
   interactions. One must assume classical spins (S=5/2) and the number of
   sites connected to a particular atom by this exchange interaction is eight.
   
We will now perform a more systematic evaluation of the exchange constants
and resulting magnon dispersion. These can be calculated from DFT using the
so-called magnetic force theorem (MFT) [1]_. The present implementation is
based on plane waves and for technical reasons it is easiest to evaluate the
Heisenberg parameters in `q`-space [2]_. Thus we obtain `J^{ab}(\mathbf{q})`,
but this is often convenient since these functions are precisely
what is required to calculate the magnon dispersion. The script
:download:`mft_q.py` computes the Fourier transformed exchange
constants and calculates the magnon dispersion along a specified path. Run
the calculation, it will take 4-5 minutes on six cores. The results can be
plotted with :download:`plot_mft.py`.

 * How does the magnon band width compare with the experimental one [3]_?

 * Identify the path segments where the degeneracy between the two bands is
   lifted by the altermagnetic (lack of) symmetries. What is the magnitude
   of magnon splitting? You may want to zoom in on particular region, which
   can be done interactively in the window.

The magnon dispersion is sampled very roughly. This is because we can only
sample `q`-points, which are on the original `4\times4\times4` regular
`k`-point mesh. The computational load quickly becomes intractable if we
increase the `k`-point sampling and we will not attempt that here. On the
other hand, if we know the exchange constants in real space we can perform
the calculation using any value of `\mathbf{q}` and obtain a magnon dispersion
with much better resolution. The exchange constants in real space will also
enable us to analyse which interactions induce the magnon splitting.  The
script :download:`mft_allbz.py` computes `J^{ab}(\mathbf{q})` on the entire
`4\times4\times4` mesh. This will allow us Fourier transform the results and
get all interactions in a range of 4 unit cells. The script will take 15-20
minutes on 6 cores. While waiting you may try to calculate and plot the magnon
dispersion resulting from a single inter-sublattice `J` as estimated from
two single DFT calculations (FM and AFM) above. You can, for example, show it
together with the MFT calculation obtained from ``mft_q.py`` and plotted with
``plot_mft.py``. Once the
calculation has finished we can extract specific exchange constants with
the script :download:`get_Jij.py`. This essentially calculates:

.. math:: J^{ab}_{0i}=\frac{1}{N_\mathbf{q}}\sum_\mathbf{q}
	  J^{ab}(\mathbf{q})e^{i\mathbf{q}\cdot\mathbf{R}_i}

where both `\mathbf{R}_i` and `\mathbf{q}` are given in reduced coordinates
in the script. Inspect the output

 * What is the largest exchange constant? Is the nearest neighbour one?

 * How does the estimate from the two DFT calculations (FM and AFM) above
   compare to the present results?

 * Can you identify the symmetry breaking exchange paths?

We can be a bit more systematic about this and loop over all unit cells.
The script :download:`get_allJ.py` does exactly that and sorts the
results according to exchange distances. It should be clear from the output
that the altermagnetic symmetry breaking does not enter before the seventh
nearest neighbour interaction.

Now that we have all the exchange constants, we may calculate a refined
magnon dispersion.

 * Write a script that plots the magnon dispersion of ``plot_mft.py``
   on a 10-fold increased resolution. Hint: generate a path from 
   ``mft_q.py`` using  ``kpts = 40``. Use the points on the path
   to calculate the Fourier transform of `J^{ab}_{0i}`, which was
   calculated in ``get_allJ.py``. The resulting `J^{ab}(\mathbf{q})`
   can be used to get the magnon dispersion as was done in
   ``plot_mft.py``.
   
.. [1] *A.I. Liechtenstein, M.I. Katsnelson, V.P. Antropov,
       and V.A. Gubanov*, JMMM **67**, 65 (1987)

.. [2] *F. L. Durhuus, T. Skovhus, and T. Olsen*, J. Phys.: Cond. Mat.
       **35**, 105802 (2023)

.. [3] *A. Okazaki, K. Turberfield, and R. Stevenson* Phys. Lett.
       **8**, 9 (1964)
