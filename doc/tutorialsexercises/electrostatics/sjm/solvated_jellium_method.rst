.. _solvated_jellium_method:

=============================
Solvated Jellium Method (SJM)
=============================

Please familiarize yourself with the theory of the SJM approach before starting this tutorial, by reading the :ref:`SJM documentation page<sjm>`.

A simple Au(111) slab
=====================

As a simple example, we'll examine the calculation of a Au slab with a water overlayer calculated at a potential of -0.2 V versus SHE.
To speed the computation, we'll use a tiny slab with a single water molecule over the top.

To use the solvated jellium approach, instead of importing GPAW as our calculator object, we import :class:`~gpaw.solvation.sjm.SJM`.
All of the SJM-specific parameters are fed to the calculator as a single dictionary called :literal:`sj`.
In our example script, the :literal:`sj` dictionary contains only a single parameter: the target potential; the rest of the SJM parameters are kept at their default values.
Keep in mind that the *absolute* potential has to be provided, where the value of the SHE potential on an absolute scale is approximately 4.4 V.

Since SJM utilizes an implicit solvent to screen the field, we will need to specify some solvent parameters as well.
SJM relies on GPAW's :ref:`continuum_solvent_model` to provide this screening, and any solvation parameters should be fed directly to the calculator, *outside* of the :literal:`sj` dictionary.
(The SJM calculator is a subclass of the :class:`~gpaw.solvation.calculator.SolvationGPAW` calculator, so any solvation parameters will be passed straight through to SolvationGPAW.)
Reasonable values to simulate room-temperature water are used in the script below.
In practice, since the purpose of the implicit solvent is just to screen the field, the net results---such as binding energies and barrier heights---will be relatively insensitive to the choice of implicit solvent parameters, so long as they are kept consistent.
Try running the script below to find the number of electrons necessary to equilibrate the work function to 4.2 eV:

.. literalinclude:: run-Au111.py

If you examine the output in 'Au111.txt', you'll see that the code varies the number of excess electrons until the target potential is within a tolerance of 4.2 V; the default tolerance is 0.01 V.
This process usually takes a few steps on the first image, but is faster on subsequent images of a trajectory, since the changes are less dramatic and the potential--vs--charge slope is retained from previous steps.
You should see that in net the routine added only about 0.007 electrons to the simulation cell, as compared to a charge-neutral simulation, in order to achieve the desired work function.

You'll notice that the output in 'Au.txt' contains additional information, as compared to a charge-neutral simulation::

    Legendre-transformed energies (grand potential, Omega = E - N mu)
     N (excess electrons):    +0.006526
     mu (-workfunction, eV):   -4.208990
     (Grand) free energy:    -23.628448
     (Grand) extrapolated:   -23.605878

These Legendre-transformed energies, `\Omega = E - N \mu`, are written into any ASE trajectories created, and are the quantity returned by :literal:`calc.get_potential_energy()` (and therefore :literal:`atoms.get_potential_energy()`).
As discussed in :ref:`grand-potential-energy`, these grand-potential energies are consistent with the forces in the grand-canonical scheme, and are thus compatible with methods such as saddle-point searches (*e.g.*, NEB) and energy optimizations (*e.g.*, BFGSLineSearch).
If you'd rather have it output the traditional canonical energies to the :literal:`get_potential_energy` methods, you can use the :literal:`sj['grand_output']` keyword.

In the last line of the script above, we wrote out some additional information in the form of 'sjm traces'.
These are traces of the electrostatic potential, the background charge, and the solvent cavity across the `z` axis (that is, `xy`-averaged).
It's a good idea to take a look at these, and perhaps make a plot for each, when running a simulation with a new system so you can be sure it is behaving as you expect.
The three files that are created are:

potential.txt:
 Electrostatic potential averaged over `xy` and referenced to the system's Fermi Level.
 The outer parts should correspond to the respective work functions and the potential at the top boundary should correspond to the potential set in the input.

cavity.txt:
 The shape of the implicit solvent cavity averaged over `xy`.
 Multiplying the cavity function with epsinf corresponds to the `xy`-averaged dielectric constant distribution in the unit cell.

background_charge.txt:
 The shape of the jellium background charge averaged over `xy` and normalized.

.. Note:: Alternatively, :literal:`calc.write_sjm_traces(style='cube')`
          can be used to write cube files on the 3-D grid, instead of traces.

Our traces showing the solvent region and the jellium region are plotted below.
The black dots indiate the atom's `z` positions.
We see that both the solvent and the jellium are starting above the highest atom, so all is good.
If you do see solvent appearing in a region where you do not want it, you can block it with ghost atoms or a boundary plane, as described in the :class:`~gpaw.solvation.sjm.SJMPower12Potential` docstring.
If you see solvent in the metal region, you could increase the atomic radii as described in the same docstring.

.. image:: traces-Au111.png
           :width: 600 px
           :align: center

(The script used to generate this plot is here: :download:`plot-traces.py`.)

Relaxing the slab and adsorbate
===============================

Next, let's perform a structural optimization at a constant potential.
Use the same system as the previous example, but add an H atom to the surface as an adsorbate.

.. literalinclude:: run-Au111-H-seq.py

If you examine the output, you should see this behaves the same as any other relaxation job, except the potential is constant, rather than the number of electrons.

A faster route: simultaneous potential and structure optimization
=================================================================

It is often faster to optimize the structure and the potential simultaneously.
That is, instead of waiting until the potential is perfectly equilibrated before taking an ionic step, you can take an ionic step and adjust the number of electrons simultaneously.
To do this, set a loose potential tolerance and turn the :literal:`always_adjust` keyword to :literal:`True`.
An example follows.

.. literalinclude:: run-Au111-H-sim.py

Some things to note:

* At the end of the script, we tightened the criteria back up, in order to guarantee precise results consistent with a normal, sequential calculation (assuming it found the same local minimum).

* It's a good idea to use the BFGS, and not the BFGSLineSearch, algorithm for structural optimization. This is because the forces and energy are not necessarily consistent until the optimization finishes, and that might confuse an optimizer that uses the energy. BFGSLineSearch may in general have trouble with SJM, since the tolerance set on the desired potential can lead to small inconsistencies between the forces and energy. When in doubt, use the traditional BFGS optimizer.

The plot below compares the "sequential" and "simultaneuous" optimization approaches.
The open circles are potential-equilibration steps, while the filled circles represent ionic steps (in which the atoms are moved).
In sequential mode, it always makes sure the potential is within tolerance before taking an ionic step, and you can see that it typically takes an extra DFT calculation (or two) before moving the atoms.
In simultaneous mode, it moves the atoms with much higher frequency, and only when the potential gets very far from the target does it pause to equilibrate.
Generally, this results in faster total convergence.
At the end of the sequential mode, we switched back to simultaneous mode to ensure that the final result was at the expected potential; this adds a few extra steps, but overall the computational savings are typically worth it.

.. image:: simultaneous.png
           :width: 600 px
           :align: center

(Plot script: :download:`plot-seq-sim.py`.)

Finding a barrier (NEB/DyNEB)
=============================

Perhaps the most-desired use of constant-potential calculations is to find barriers at a specified potential.
This avoids a well-known problem in canonical electronic structure calculations, where the work function (that is, the electrical potential) can change by 1--2 V over the course of an elementary step.
Such a potential change is obviously much greater than the experimental situation, where potentiostats hold the potential to a tolerance many orders of magnitude smaller.
The SJM method allows one to calculate a reaction barrier in a manner where all images in the trajectory have identical work functions, down to a user-specified tolerance.

The discrepancy between a canonical and grand-canonical simulation is most pronounced in reactions involving the creation or destruction of an ion, as the ion collides with the constant-potential surface.
Unfortunately, simulating even a simple reaction like the Volmer reaction---where a proton in the solution reacts to form an adsorbed hydrogen atom---is not possible in this tutorial, as a relatively large water layer---that is, larger than the one molecule we are using!---is needed in order to stabilize the solvated proton in the initial state.
Therefore, in this tutorial, we'll do a much simpler reaction: the diffusion of our adsorbed H atom from one type of surface site to another.
Note that we should expect very little difference in the number of excess electrons in order to equilibrate the initial, transition, and final state; nevertheless, it allows us to demonstrate the use of the nudged elastic band (NEB) method with SJM.

You will need to create the final state before you run the NEB; you can go through the exercise of creating this yourself, or you can use our script (:download:`run-Au111-H-hollow.py`).
After you have created the final state, you can run the NEB as below; note here we use the DyNEB method which tends to save a large amount of computational time for these sorts of barriers.

.. literalinclude:: run-neb.py

After a few iterations, the band completes and we can make a plot of the barrier, using :literal:`NEBTools`.
(Plot script: :download:`plot-neb.py`.)

.. image:: band.png
           :width: 600 px
           :align: center

Although the reaction modeled here is very simple and doesn't involve much need for a constant-potential method, it nonetheless allows us to demonstrate that SJM plays well with standard barrier-search methods, like the NEB.

You'll note that we used :literal:`DyNEB` instead of the regular :literal:`NEB`.
Although :literal:`NEB` will also work fine, :literal:`DyNEB` is more efficient as it skips re-calculating individual images whose maximum force is already below the cutoff (of :literal:`fmax=0.05`).
Note also that parallelizing a NEB over images is in general not a great strategy when using a grand-canonical method, because typically only one of the images might need potential equilibration on any individual force call of the NEB---and in parallel-image mode all of the images would need to wait for that individual image.
However, running in serial allows us to take advantage of the computational efficiency of the :literal:`DyNEB` method, which is not possible when parallelizing over images.

Constant-charge mode
====================

The SJM code can also run in constant-charge mode, where the user specifiies the total number of electrons in the simulation.
This can be a fast way to calculate a system at several potentials---that is, if one does not care about the specific potentials, but just wants to span a range of potentials.
To use constant-charge mode, just specify the number of :literal:`excess_electrons` in your :literal:`sj` dict and calculate as normal::

    sj = {'excess_electrons': 0.5}
    calc = SJM(sj=sj, ...)


If your intent is to run in the constant-charge ensemble---like in the use of the charge-extrapolation scheme---you may want to output the canonical, rather than the grand-potential energies.
The canonical energies are consistent with the forces in constant-charge mode.
To accomplish this, set :literal:`sj['grand_output'] = False`, like::

    sj = {'excess_electrons': ...,
          'grand_output': False} :
    calc = SJM(sj=sj, ...)
