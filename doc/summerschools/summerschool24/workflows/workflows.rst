=======================
Computational workflows
=======================

In this exercise we will write and run computational workflows
with `TaskBlaster <https://taskblaster.readthedocs.io/en/latest/>`_.

The basic unit of computation in TaskBlaster is a *task*.  A task is a Python
function along with a specification of the inputs to that function.
The inputs can be either concrete values like lists, strings, or numbers,
or references to the outputs of other tasks.
Tasks that depend on one another and form a graph.  A
TaskBlaster workflow is a Python class which defines such a graph, along with
metadata about how the tasks should run and be stored.  Workflows can
then be parametrized so they run on a collection of materials,
for example.

When using TaskBlaster, we define workflows and tasks using Python.
However, the tools used to *run* workflows and tasks are command-line
tools.  Therefore, for this exercise we will be using the terminal
rather than the usual notebooks.  Basic knowledge of shell commands
is an advantage.

This exercise consists of multiple parts:

 * Introductory tutorials to TaskBlaster
 * Write a workflow which defines a structure optimization task
   and phonon computation using the simple EMT force field
 * Parametrize the workflow to comprise multiple materials
 * Adapt the workflow to run with GPAW, adding tasks for ground-state
   and electronic band structure
 * Submit multiple workers to execute the computational tasks at scale

When actually using ASR, many tasks and workflows are already written.
Thus, we would be able to import and use those features directly.
But in this tutorial we write everything from scratch.

Part 0: The TaskBlaster tutorials and documentation
===================================================

TaskBlaster comes with its own documentation and tutorials.
These are not related to physics or materials, but provide a generic
introduction to the framework:

 * To get started, go through the basic
   `TaskBlaster tutorials <https://taskblaster.readthedocs.io/en/latest/tutorial/module.html>`_,
   specifically
   `“Hello world” <https://taskblaster.readthedocs.io/en/latest/tutorial/hello_world/hello_world.html>`_ and
   `“My first workflow” <https://taskblaster.readthedocs.io/en/latest/tutorial/my_first_workflow/my_first_workflow.html>`_.
   Make sure you can run the examples and feel free to explore
   the command-line interface.

 * It may be illustrating to read the explanation page on
   `Basic Concepts <https://taskblaster.readthedocs.io/en/latest/explanation/basic_concepts/basic_concepts.html>`_.

We have now acquired a basic familiarity with TaskBlaster commands,
although we may still not see the full picture of how one runs
scientific high-throughput projects in practice.  We will deal with this
over the course of the next exercises.


Part 1: Simple materials workflow
=================================

In this exercise we will develop a first materials workflow using the
GPAW electronic structure code.
This exercise will be more open-ended.
Be sure to make good use of the different command-line tools'
:option:`--help` pages as well as TaskBlaster's online documentation.

TaskBlaster itself does not know anything about electronic structure,
materials, atoms, or ASE.
However we will want tasks to have ASE objects as input
and output, and that requires being able to save those objects.
TaskBlaster can be extended with the ability to encode and decode arbitrary
objects, and doing so requires a plugin.
Such a plugin is provided by asr-lib, the
`Atomic Simulation Recipes <https://gitlab.com/asr-dev/asr-lib>` library.
We will create a project using asr-lib as a plugin and so facilitate
our work with ASE objects.

Go to a clean directory and create repository using the ``asrlib`` plugin::

  $ tb init asrlib

You can use the ``tb info`` command to see global information about
the repository and verify that it uses ``asrlib``.

Set up structure optimization
-----------------------------

Write a workflow class called ``MaterialsWorkflow`` which:

 * takes ``atoms`` as an input variable
 * takes ``calculator`` as an input variable, which is a dictionary
   of keyword arguments that will be passed to GPAW
 * defines a ``relax`` task which performs a structure optimization
   that includes optimising the unit cell.

The wise scientist first writes a relax task which only prints the atoms.
That is enough to verify that the workflow works, and that the atoms
are passed and encoded correctly.  After that, unrun, edit, rerun, and fix it
until it works.

The workflow should function correctly when called on a bulk silicon system
like this:

.. literalinclude:: workflow.py
   :start-at: def workflow
   :end-before: end-workflow-function-snippet

Here we have chosen some generic GPAW parameters that will be fine
for testing, but not for production.

The relaxation task can be implemented like this:

.. literalinclude:: tasks.py
   :start-at: def optimize_cell
   :end-before: end-optimize-cell-snippet

Users unfamiliar with ASE may want to take a while to look up
ASE concepts like atoms and calculators.  What the function does is:

 * Attach a GPAW calculator to the atoms
 * Create a
   `Frechet cell filter
   <https://wiki.fysik.dtu.dk/ase/ase/filters.html#the-frechetcellfilter-class>`_
   which exposes the cell degrees
   of freedom and stresses to an optimizer
 * Run a `BFGS <https://wiki.fysik.dtu.dk/ase/ase/optimize.html#bfgs>`_
   optimization algorithm on the Frechet cell filter.

It also tells the optimizer to write a trajectory file, ``opt.traj``.

Once everything works and you run the relaxation task,
go to the task directory and use the ASE GUI (e.g. ``ase gui opt.traj``)
to visualize the trajectory.

GPAW also writes a log file to ``gpaw.txt``.
It is wise to have a brief look and observe that
the calculation used the parameters we expect.


Part 2: Add ground state and band structure tasks
=================================================

After the relaxation, we want to run a ground state
calculation to save a ``.gpw`` file, which we subsequently want
to pass to a non self-consistent calculation to get the band structure.

Add a ``groundstate()`` function to ``tasks.py``:

.. literalinclude:: tasks.py
   :pyobject: groundstate

In order to "return" the gpw file, we actually return a ``Path`` object
pointing to it.  When passing the path to another task, TaskBlaster
resolves it with respect to the task's own directory such
that the human will not need to remember or care about the actual directories
where the tasks run.

Next, add a ``groundstate()`` task to the workflow which calls the groundstate
function just added to ``tasks.py``.
By calling ``tb.node(..., atoms=self.relax)``, we are specifying
that the atoms should be taken as the *output* of the ``relax`` task,
creating a dependency.

We can now run the workflow again.  The old task still exists and
will remain unchanged, whereas the new task should now appear
in the ``tree/groundstate`` directory.

Run the ground state task and check that the ``.gpw`` file was created as
expected.

Finally, we write a band structure task in ``tasks.py``:

.. literalinclude:: tasks.py
   :pyobject: bandstructure

A corresponding method should be added on the workflow:

.. literalinclude:: workflow.py
   :pyobject: MaterialsWorkflow.bandstructure

Now run the workflow and the resulting tasks.
The code saves the Brillouin zone path and band structure separately to
ASE JSON files.  Once it runs, we can go to the directory and check
that it looks correct::

  ase reciprocal tree/bandstructure/bandpath.json

::

   ase band-structure tree/bandstructure/bs.json



You can delete all the tasks with ``tb remove tree/`` and run them from
scratch by ``tb run tree/``, ``tb run tree/*``, or simply ``tb run
tree/bandstructure``.
The run command always executes tasks in
topological order, i.e., each task runs only when its dependencies
are done.

The ``tb ls`` command can also be used to list tasks in topological
order following the dependency graph::

  human@computer:~/myworkflow$ tb ls --parents tree/bandstructure/

  state    deps  tags        worker        time     folder
  ───────────────────────────────────────────────────────────────────────────────
  done     0/0               N/A-0/1       00:00:04 tree/relax
  done     1/1               N/A-0/1       00:00:01 tree/groundstate
  done     1/1               N/A-0/1       00:00:04 tree/bandstructure


This way, we can comfortably work with larger numbers of tasks.

If we edit the workflow such that tasks receive different inputs,
TaskBlaster will mark the affected tasks with a conflict.
Such a conflict can be solved by removing the old calculations
or by telling TaskBlaster to consider it “resolved”, which we
have seen in a previous tutorial.


TODO and old stuff
------------------


Phonon computation
------------------

Now that we have the optimized structure, we can perform a phonon
calculation on top.  We will use the ASE phonons module for this.
Getting this to work with the EMT force field is a nice first step
that can later be used to make a workflow that uses a real electronic
structure code such as GPAW.

It is generally desirable to separate computations into distinct chunks:
For example the phonon computation involves force calculations
on displaced atoms, which would be expensive, and those should go
in one task.  Postprocessing on top of that via the dynamical matrix,
including computation of phonon band structure, should go on top of that.

Unfortunately, the ASE phonon implementation makes such a separation
difficult, so we provide snippets with hacks that make it work.



Let's perform a structure optimization of bulk Si.
We write a function which performs such an optimization:

..
   literalinclude:: tasks.py
   :end-before: end-snippet-1

This function uses a cell filter to expose the cell degrees of freedom
for the standard BFGS optimizer (see the ASE documentation on optimizers
and cell filters if interested).

Since workflows run on the local computer whereas computational tasks
(generally) run on compute nodes, we separate *workflow* code
and *computational*
code in different files.  ASR can load user-defined functions from the
special file ``tasks.py`` mentioned by info command.
Create that file and save the above function to it.

Next, we write a workflow with a task that will call the function:

..
   literalinclude:: workflow.py
   :end-before: end-snippet-1

Explanation:

* The ``@asr.workflow`` decorator tells ASR to regard the class as a
  workflow.  In particular, it equips the class with a constructor
  with appropriate input arguments.

* ``asr.var()`` is used to declare input variables.  The names
  ``atoms`` and ``calculator`` imply
  that we want this workflow to take atoms and calculator parameters
  as input.

* The method ``relax()`` defines our task.
  By naming the method ``relax()``, we choose that the task will run
  in a directory called ``tree/relax``.

* The method returns ``asr.node(...)``, which is a specification of
  the *actual* calculation: The name of the task
  (``'relax'``, which must exist in ``tasks.py``) is given as a string.
  The inputs are then assigned, and will be forwarded to the
  ``relax()`` function in ``tasks.py``.
  The attributes ``self.atoms`` and ``self.calculator``
  will refer to the input variables.

  When defining a node, ASR calculates a hash (i.e. checksum) of the inputs;
  the hash will become different if any inputs are changed.

* The decorator ``@asr.task`` can be used to attach information
  about *how* the task runs, such as computational
  resources.


The workflow class serves as a *static declaration* of information, not as
statements or commands to be executed (yet).
To actually run it, we must at least choose a material and then tell
the computer to run the workflow on it.
We do this by adding a standalone function called ``workflow``
for ASR to call:

.. literalinclude:: workflow.py
   :pyobject: workflow


ASR will take care of creating a "runner" and passing it to the function.
(Note: In a future version of the code, this syntax will be simplified.)


Save the code (both the class and the ``workflow()`` function)
to a file, e.g. named ``workflow.py``.  Then execute the
workflow by issuing the command::

  asr workflow workflow.py

The command executes the workflow and creates a folder under the ``tree/``
directory for each task.
We can run ``asr ls`` to see a list of the tasks we generated::

  541d427c new      tree/relax                     relax(atoms=…, calculator=…)

The task is identified to the computer as a hash value (541d427c....),
whereas to a human user, the location in the directory tree,
``tree/relax``, will be more descriptive.

Feel free to look at the contents of the ``tree/relax`` directory.
The task is listed as "new" because we did not run it yet
— we only *created* it, so far.  While developing workflows,
we will often want to
create and inspect the tasks before we submit anything expensive.
If we made a mistake, we can remove the task with
``asr remove tree/relax``, then fix the mistake and run the workflow again.

Once we're happy with the task, let's run the task on the local computer::

  asr run tree/relax

If everything worked as intended, the task will now be "done",
which we can see by running ``asr ls`` again::

  541d427c done     tree/relax                     relax(atoms=…, calculator=…)


We can use the very handy ``tree`` command to see the whole
directory tree::

  human@computer:~/myworkflow$ tree tree/
  tree/
  └── relax
      ├── gpaw.txt
      ├── input.json
      ├── input.resolved.json
      ├── opt.log
      ├── opt.traj
      ├── output.json
      └── state.dat

  1 directory, 7 files

Be sure to open the trajectory file in (e.g. in ASE GUI) to check
that the optimization ran as expected.  Also the logfiles
``gpaw.txt`` and ``opt.log`` are there.




Part 3: Run workflow on multiple materials
==========================================

The current workflow creates directories right under the repository root.
For a proper materials workflow, it will be helpful to work
with a structure that nests the tasks by material.

ASR contains a feature called ``totree`` which deploys a dataset
to the tree, such as defining initial structures for materials.
One then parametrizes a workflow (such as the one we just wrote)
on the materials.

The following workflow defines a function which returns a set of materials,
then specifies to ASR that those must be added to the tree.

.. literalinclude:: totree.py

Add this to a new file, named e.g. ``totree.py``, and execute the workflow::

 human@computer:~/myworkflow$ asr workflow totree.py
       Add: 889575c5 new      tree/Al/material               define(obj=…)
       Add: 5e39fb8e new      tree/Si/material               define(obj=…)
       Add: 9612a07a new      tree/Ti/material               define(obj=…)
       Add: 7153df81 new      tree/Cu/material               define(obj=…)
       Add: 155d59ee new      tree/Ag/material               define(obj=…)
       Add: e9b41657 new      tree/Au/material               define(obj=…)

The totree command created some tasks for us.
Actually they are not really tasks — they are just static pieces of data.
But now that they exist, we can run other tasks that depend on them.

In the old workflow file (``workflow.py``),
replace the ``workflow()`` function with the following function which
tells ASR to parametrize the workflow by "globbing" over the materials:


.. literalinclude:: materials.py
   :pyobject: workflow

The workflow will now be called once for each material.
Run the workflow and it will create our three well-known tasks
for each material, now nested by material.

As before, we can inspect the newly created tasks, e.g.::

 human@computer:~/myworkflow$ asr ls tree/Au/bandstructure/ --parents
 e9b41657 new      tree/Au/material               define(obj=…)
 5306d226 new      tree/Au/relax                  relax(atoms=<e9b41657>, calculator=…)
 a54f98a7 new      tree/Au/groundstate            groundstate(atoms=<5306d226>, calculator=…)
 7fbfa099 new      tree/Au/bandstructure          bandstructure(gpw=<a54f98a7>)


Since it may take a while to run on the front-end node,
we can tell ASR to submit one or more tasks using MyQueue_::

  asr submit tree/Au

The submit command works much like the run command, only it calls
myqueue which will then talk to the scheduler (slurm, torque, ...).
After submitting, we can use standard myqueue commands to monitor
the jobs, such as ``mq ls`` or ``mq rm``.  See the `myqueue documentation
<https://myqueue.readthedocs.io/en/latest/cli.html>`_.

If everything works well, we can submit the whole tree::

  asr submit tree/

Note: In the current version, myqueue and ASR do not perfectly
share the state of a task.  This can lead to
mild misbehaviours if using both ``asr run`` and ``asr submit``,
such as a job executing twice.


.. _MyQueue: https://myqueue.readthedocs.io/
