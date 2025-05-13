==============================
Introduction to GPAW internals
==============================

.. testsetup::

    from gpaw.fftw import *
    from gpaw.core.matrix import *
    from gpaw.core.atom_arrays import *
    from gpaw.new.symmetry import Symmetries

.. contents::

DFT input parameters
====================
The :class:`~gpaw.dft.Parameters` object takes care of
input parameters:

* checks for errors
* does normalization
* handles backwards compatibility and deprecation warnings


>>> from gpaw.new.calculation import DFTCalculation
>>> calculation = DFTCalculation.from_parameters(atoms, params)

or when you create an ASE-calculator interface:

>>> from ase import Atoms
>>> atoms = Atoms('Li', cell=[2, 2, 2], pbc=True)
>>> from gpaw.new.ase_interface import GPAW
>>> atoms.calc = GPAW(**params, txt='li.txt')


DFT components builders
=======================

The components needed for a DFT calculation are created by a "builder" that
can be made from an ASE
:class:`ase.Atoms` object and some input parameters.
There are builders for each of the modes: PW, FD and LCAO.


Full picture
============

The :class:`ase.Atoms` object has an
:class:`gpaw.new.ase_interface.ASECalculator` object attached
created with the :func:`gpaw.new.ase_interface.GPAW` function:

>>> atoms = Atoms('H2',
...               positions=[(0, 0, 0), (0, 0, 0.75)],
...               cell=[2, 2, 3],
...               pbc=True)
>>> atoms.calc = GPAW(mode='pw', txt='h2.txt')
>>> atoms.calc
ASECalculator(mode: {'name': 'pw'})

The ``atoms.calc`` object manages a
:class:`gpaw.new.calculation.DFTCalculation` object that does the actual work.
When we do this:

>>> e = atoms.get_potential_energy()

the :meth:`gpaw.new.ase_interface.ASECalculator.get_potential_energy`
method gets called (``atoms.calc.get_potential_energy(atoms)``)
and the following will happen:

* create :class:`gpaw.new.calculation.DFTCalculation` object if not already done
* update positions/unit cell if they have changed
* start SCF loop and converge if needed
* calculate energy
* store a copy of the atoms


DFT-calculation object
======================

.. module:: gpaw.core

An instance of the :class:`gpaw.new.calculation.DFTCalculation` class has
the following attributes:

.. list-table::

  * - ``density``
    - :class:`gpaw.new.density.Density`
  * - ``ibzwfs``
    - :class:`gpaw.new.ibzwfs.IBZWaveFunctions`
  * - ``potential``
    - :class:`gpaw.new.potential.Potential`
  * - ``scf_loop``
    - :class:`gpaw.new.scf.SCFLoop`
  * - ``pot_calc``
    - :class:`gpaw.new.pot_calc.PotentialCalculator`

Overview:

* ``atoms.calc``:

  * ``dft``:

    * ``density``: ``nt_sR``, ``D_asii``, ...
    * ``potential``: ``vt_sR``, ``dH_asii``, ...
    * ``pot_calc``:

      * ``xc``
      * ``poisson_solver``

    * ``scf_loop``:

      * ``eigensolver``
      * ``hamiltonian``
      * ``occ_calc``
      * ``mixer``

    * ``ibzwfs``:

      * ``ibz``:

        * ``symmetries``
        * ``bz``

      * ``wfs_qs[q][s]``:

        * ``psit_nX``
        * ``occ_n``
        * ...

See also: :download:`code.svg`.


Naming convention for arrays
============================

Commonly used indices:

 =======  ====================================================================
 index    description
 =======  ====================================================================
 ``a``    Atom number
 ``c``    Unit cell axis-index (0, 1, 2)
 ``v``    *xyz*-index (0, 1, 2)
 ``K``    BZ **k**-point index
 ``k``    IBZ **k**-point index
 ``q``    IBZ **k**-point index (local, i.e. it starts at 0 on each processor)
 ``s``    Spin index (`\sigma`)
 ``s``    Symmetry index
 ``u``    Combined spin and **k**-point index (local)
 ``R``    Three indices into the coarse 3D grid
 ``r``    Three indices into the fine 3D grid
 ``G``    Index of plane-wave coefficient (wave function expansion, ``ecut``)
 ``g``    Index of plane-wave coefficient (densities, ``2 * ecut``)
 ``h``    Index of plane-wave coefficient (compensation charges, ``8 * ecut``)
 ``X``    ``R`` or ``G``
 ``x``    ``r``, ``g`` or ``h``
 ``x``    Zero or more extra dimensions
 ``M``    LCAO orbital index (`\mu`)
 ``n``    Band number
 ``n``    Principal quantum number
 ``l``    Angular momentum quantum number (s, p, d, ...)
 ``m``    Magnetic quantum number (0, 1, ..., 2*`ell` - 1)
 ``L``    ``l`` and ``m`` (``L = l**2 + m``)
 ``j``    Valence orbital number (``n`` and ``l``)
 ``i``    Valence orbital number (``n``, ``l`` and ``m``)
 ``q``    ``j1`` and ``j2`` pair
 ``p``    ``i1`` and ``i2`` pair
 ``r``    CPU-rank
 =======  ====================================================================

Examples:

.. list-table::

  * - ``density.D_asii``
    - `D_{\sigma,i_1,i_2}^a`
    - :class:`~atom_arrays.AtomArrays`
  * - ``density.nt_sR``
    - `\tilde{n}_\sigma(\mathbf{r})`
    - :class:`~UGArray`
  * - ``ibzwfs.wfs_qs[q][s].P_ani``
    - `P_{\sigma \mathbf{k} ni}^a`
    - :class:`~atom_arrays.AtomArrays`
  * - ``ibzwfs.wfs_qs[q][s].psit_nX``
    - `\tilde{\psi}_{\sigma \mathbf{k} n}(\mathbf{r})`
    - :class:`~UGArray` |
      :class:`~PWArray`
  * - ``ibzwfs.wfs_qs[q][s].pt_aX``
    - `\tilde{p}_{\sigma \mathbf{k} i}^a(\mathbf{r}-\mathbf{R}^a)`
    - :class:`~atom_centered_functions.AtomCenteredFunctions`


Domain descriptors
==================

GPAW has two different container types for storing one or more functions
in a unit cell (wave functions, electron densities, ...):

* :class:`~PWArray`
* :class:`UGArray`


Uniform grids
-------------

A uniform grid can be created with the :class:`UGDesc` class:

>>> import numpy as np
>>> from gpaw.core import UGDesc
>>> a = 4.0
>>> n = 20
>>> grid = UGDesc(cell=a * np.eye(3),
...               size=(n, n, n))

Given a :class:`UGDesc` object, one can create
:class:`UGArray` objects like this

>>> func_R = grid.empty()
>>> func_R.data.shape
(20, 20, 20)
>>> func_R.data[:] = 1.0
>>> grid.zeros((3, 2)).data.shape
(3, 2, 20, 20, 20)

Here are the methods of the :class:`UGDesc` class:

.. csv-table::
   :file: ugd.csv

and the :class:`UGArray` class:

.. csv-table::
   :file: uga.csv


Plane waves
-----------

A set of plane-waves are characterized by a cutoff energy and a uniform
grid:

>>> from gpaw.core import PWDesc
>>> pw = PWDesc(ecut=100, cell=grid.cell)
>>> func_G = pw.empty()
>>> func_R.fft(out=func_G)
PWArray(pw=PWDesc(ecut=100 <coefs=1536/1536>, cell=[4.0, 4.0, 4.0], pbc=[True, True, True], comm=0/1, dtype=float64), dims=())
>>> G = pw.reciprocal_vectors()
>>> G.shape
(1536, 3)
>>> G[0]
array([0., 0., 0.])
>>> func_G.data[0]
np.complex128(1+0j)
>>> func_G.ifft(out=func_R)
UGArray(grid=UGDesc(size=[20, 20, 20], cell=[4.0, 4.0, 4.0], pbc=[True, True, True], comm=0/1, dtype=float64), dims=())
>>> round(func_R.data[0, 0, 0], 15)
np.float64(1.0)

Here are the methods of the :class:`~PWDesc` class:

.. csv-table::
   :file: pwd.csv

and the :class:`~PWArray` class:

.. csv-table::
   :file: pwa.csv


Atoms-arrays
============

As an example, here is how to store the PAW atomic density-matrices for
a water molcule
(:math:`D_{\sigma,i_1,i_2}^a`):

.. code-block:: python

   >>> nspins = 2
   >>> D_asii = AtomArraysLayout(
   ...     [(5, 5), (5, 5), (13, 13)],
   ...     # dtype=float,
   ...     # xp=np,
   ...     # atomdist=...
   ...     ).zeros(nspins)
   >>> D_asii.data.shape
   (2, 219)


Matrix elements
===============

>>> psit_nG = pw.zeros(5)
>>> def T(psit_nG):
...     """Kinetic energy operator."""
...     out = psit_nG.new()
...     out.data[:] = psit_nG.desc.ekin_G * psit_nG.data
...     return out
>>> H_nn = psit_nG.matrix_elements(psit_nG, function=T)

Same as:

>>> Tpsit_nG = T(psit_nG)
>>> psit_nG.matrix_elements(Tpsit_nG, symmetric=True)
Matrix(float64: 5x5)

but faster.


Atom-centered functions
=======================

.. literalinclude:: acf_example.py

.. figure:: acf_example.png


Matrix object
=============

.. module:: gpaw.core.matrix

Here are the methods of the :class:`~Matrix` class:

.. csv-table::
   :file: m.csv

A simple example that we can run with MPI on 4 cores::

    from gpaw.core.matrix import Matrix
    from gpaw.mpi import world
    a = Matrix(5, 5, dist=(world, 2, 2, 2))
    a.data[:] = world.rank
    print(world.rank, a.data.shape)

Here, we have created a 5x5 :class:`Matrix` of floats distributed on a 2x2
BLACS grid with a block size of 2 and we then print the shapes of the ndarrays,
which looks like this (in random order)::

    1 (2, 3)
    2 (3, 2)
    3 (2, 2)
    0 (3, 3)

Let's create a new matrix ``b`` and :meth:`redistribute <Matrix.redist>`
from
``a`` to ``b``::

    b = a.new(dist=(None, 1, 1, None))
    a.redist(b)
    if world.rank == 0:
        print(b.array)

This will output::

    [[ 0.  0.  2.  2.  0.]
     [ 0.  0.  2.  2.  0.]
     [ 1.  1.  3.  3.  1.]
     [ 1.  1.  3.  3.  1.]
     [ 0.  0.  2.  2.  0.]]

Matrix-matrix multiplication
works like this::

    c = a.multiply(a, opb='T')


API
===

Core
----

.. autoclass:: gpaw.core.UGDesc
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.PWDesc
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.atom_centered_functions.AtomCenteredFunctions
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.UGArray
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.arrays.DistributedArrays
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.atom_arrays.AtomArrays
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.atom_arrays.AtomArraysLayout
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.atom_arrays.AtomDistribution
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.PWArray
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.plane_waves.Empty
    :members:
    :undoc-members:

.. autoclass:: Matrix
   :members:
   :undoc-members:
.. autoclass:: MatrixDistribution
   :members:
   :undoc-members:

.. autoclass:: gpaw.new.symmetry.Symmetries
   :members:
   :undoc-members:
.. autoclass:: gpaw.new.brillouin.IBZ
   :members:
   :undoc-members:
.. autoclass:: gpaw.new.brillouin.BZPoints
   :members:
   :undoc-members:
.. autoclass:: gpaw.new.brillouin.MonkhorstPackKPoints
   :members:
   :undoc-members:

.. autoclass:: gpaw.new.environment.Environment
   :members:
   :undoc-members:


DFT
---

.. autoclass:: gpaw.new.calculation.DFTCalculation
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.calculation.DFTState
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.density.Density
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.ibzwfs.IBZWaveFunctions
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.potential.Potential
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.pot_calc.PotentialCalculator
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.scf.SCFLoop
    :members:
    :undoc-members:
.. autoclass:: gpaw.dft.Parameters
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.pwfd.wave_functions.PWFDWaveFunctions
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.ase_interface.ASECalculator
    :members:
    :undoc-members:
.. autofunction:: gpaw.new.ase_interface.GPAW
.. autofunction:: gpaw.new.pwfd.move_wfs.move_wave_functions


FFTW
----

.. automodule:: gpaw.fftw
   :members:


BLAS
----

.. autofunction:: gpaw.utilities.blas.mmm
.. autofunction:: gpaw.utilities.blas.rk
