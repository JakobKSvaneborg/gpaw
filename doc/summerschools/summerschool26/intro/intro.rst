.. testsetup::

    import ase.visualize as v
    v.view = lambda atoms: None

.. _intro:

========================================================
Introduction to Python and ASE and some useful libraries
========================================================

Python is a programming language that has become a de-facto standard
within scientific programming.  It is also very popular for teaching
programming and computer science.  One of the advantages of the language
is that it is relatively easy to read and understand pre-existing code.


Importing the necesary modules
==============================

A lot of helper packages for e.g.  numeric calculations, plotting,
atomic-scale simulations etc are available in Python, but you need to
"import" the module to get access to it (you also often need to install
it, but that has been taken care of here).

You can add comments to Python code by using the ``#`` character.

Some links for reference:

* `Python <https://www.python.org/>`__
* Numerical package `NumPy <https://www.numpy.org/>`__
* Scientific package `Scipy <https://www.scipy.org/>`__
* Plotting package `Matplotlib <https://matplotlib.org/>`__

>>> import matplotlib.pyplot as plt    # For nice plotting
>>> import numpy as np                 # Mathematical operations

Let's try some simple stuff.

>>> 2 + 3
5
>>> print('Hello')
Hello
>>> print('Hello' * 5)
HelloHelloHelloHelloHello
>>> # help is useful for getting help of af method
>>> help(print)
Help on built-in function print in module builtins:
<BLANKLINE>
print(*args, sep=' ', end='\n', file=None, flush=False)
    Prints the values to a stream, or to sys.stdout by default.
<BLANKLINE>
    sep
      string inserted between values, default a space.
    end
      string appended after the last value, default a newline.
    file
      a file-like object (stream); defaults to the current sys.stdout.
    flush
      whether to forcibly flush the stream.
<BLANKLINE>


Variables and data types
========================

In python, variables contain data of different types, such as numbers or
text strings.  You can print a variable with the ``print()`` function:

>>> a = 42
>>> mypi = 3.14
>>> b = 'some text'
>>> a
42
>>> print('Pi is approximately', mypi)
Pi is approximately 3.14
>>> b
'some text'

The usual mathematical operations are possible.  The operator for
exponentiation is the double star.

>>> 2 * a
84
>>> 16**2
256
>>> a + 2 * mypi
48.28


A ``list`` is an ordered collection of arbitrary objects
--------------------------------------------------------

You will need lists of data.  A list of data can be specified at once, or
built gradually.  The latter is demonstated later.

>>> primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

Lists are indexed starting with 0, so primes[1] is the *second* prime.
You can also access a list from the end, using negative numbers.

>>> primes[1]
3
>>> primes[-1]
23

A `list` can contain arbitrary objects

>>> # a list
>>> l = [1, ('gg', 7), 'hmm', 1.2]
>>> l[1]   # Python counts from zero, so this is the second element
('gg', 7)
>>> l[-2]  # indexing with negative numbers counts from the end
'hmm'


A ``dict``  is a mapping from keys to values
--------------------------------------------

>>> d = {'s': 0, 'p': 1}
>>> d
{'s': 0, 'p': 1}
>>> d['p']
1
>>> # Removing an element from the dictionary
>>> del d['s']
>>> d
{'p': 1}


A ``tuple``  is an ordered collection like a list but is *immutable*
--------------------------------------------------------------------

useful for keywords in ``dict``

>>> # with a list we can reassign values
>>> x = [2, 3]
>>> x[0] = 100
>>> x
[100, 3]
>>> # this it not possible with a tuple
>>> y = (2, 3)
>>> y[0] = 100
Traceback (most recent call last):
  File "<doctest default[4]>", line 1, in <module>
    y[0] = 100
    ~^^^
TypeError: 'tuple' object does not support item assignment


=====
NumPy
=====

NumPy arrays are heavely used in `ASE <https://ase-lib.org/>`__. ASE makes
heavy use of an extension to Python called NumPy.  The NumPy module
defines an :class:`~numpy.ndarray` type that can hold large arrays of
uniform multidimensional numeric data.  An array is similar to a ``list``
or a ``tuple``, but it is a lot more powerful and efficient.

>>> x = np.array([1, 2, 3])
>>> x
array([1, 2, 3])
>>> x.mean()
np.float64(2.0)

>>> # Multidimensional array
>>> a = np.zeros((3, 2))
>>> a[:, 1] = 1.0
>>> a[1, :] = 2.0
>>> a.shape
(3, 2)
>>> a.ndim  # number of dimensions
2
>>> a.dtype  # data type
dtype('float64')
>>> # And one can print the entire array (if it is not too big):
>>> a
array([[0., 1.],
       [2., 2.],
       [0., 1.]])

>>> # Matrix muliplication
>>> a.T.shape  # .T transpose a matrix
(2, 3)
>>> b = np.dot(a, a.T)
>>> b
array([[1., 2., 1.],
       [2., 8., 2.],
       [1., 2., 1.]])
>>> # in a more READABLE way one can use @ to dot matrices together
>>> c = a @ a.T
>>> print('c is equal to b:', (c == b).all())
c is equal to b: True
>>> # Elementwise multiplication
>>> a * a
array([[0., 1.],
       [4., 4.],
       [0., 1.]])

>>> # Random Hermitian matrix
>>> seed = 12345678
>>> rng = np.random.default_rng(seed)
>>> rand = rng.random
>>> H = rand((6, 6)) + 1j * rand((6, 6))  # 1j = sqrt(-1)
>>> H = H + H.T.conj()

>>> # Eigenvalues and eigenvectors
>>> eps, U = np.linalg.eig(H)

>>> #  Make print of numpy arrays less messy:
>>> np.set_printoptions(precision=3, suppress=True)
>>> print('The eigenvalues are:', eps.real)
The eigenvalues are: [ 6.149 -2.369  1.979 -0.692  0.258  0.93 ]

>>> # lets try and sort them
>>> sorted_indices = eps.real.argsort()
>>> eps = eps[sorted_indices]
>>> U = U[:, sorted_indices]
>>> print('after sorting: ', eps.real)
after sorting:  [-2.369 -0.692  0.258  0.93   1.979  6.149]

>>> # Check that U diagonalizes H
>>> D1 = np.diag(eps)  # Diagonal matrix
>>> D2 = U.T.conj() @ H @ U  # Diagonalized H matrix
>>> # Diagonal matrix (from eigenvalues):
>>> print(D1)  # doctest: +SKIP
[[-2.369+0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j]
 [ 0.   +0.j -0.692+0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j]
 [ 0.   +0.j  0.   +0.j  0.258-0.j  0.   +0.j  0.   +0.j  0.   +0.j]
 [ 0.   +0.j  0.   +0.j  0.   +0.j  0.93 +0.j  0.   +0.j  0.   +0.j]
 [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  1.979+0.j  0.   +0.j]
 [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  6.149+0.j]]
>>> # Diagonal matrix (tranforming H):
>>> print(D2)  # doctest: +SKIP
[[-2.369+0.j  0.   -0.j -0.   +0.j  0.   +0.j  0.   +0.j  0.   -0.j]
 [ 0.   +0.j -0.692+0.j  0.   +0.j -0.   +0.j  0.   +0.j -0.   -0.j]
 [-0.   -0.j  0.   +0.j  0.258-0.j -0.   -0.j -0.   -0.j  0.   +0.j]
 [ 0.   -0.j -0.   -0.j -0.   +0.j  0.93 +0.j  0.   -0.j -0.   +0.j]
 [ 0.   -0.j  0.   -0.j -0.   +0.j  0.   +0.j  1.979-0.j -0.   -0.j]
 [ 0.   +0.j -0.   +0.j -0.   -0.j  0.   +0.j -0.   +0.j  6.149+0.j]]
>>> # Are the numbers in the two matrices close to each other?
>>> np.allclose(D2, D1)
True


========================
Plotting with matplotlib
========================

(see here for more details `Matplotlib <https://matplotlib.org/>`__)

.. literalinclude:: mpl-demo.py
   :start-after: create
   :end-at: f1

.. image:: f1.svg

You can save the plot as a figure by pressing the "floppy disk" icon.
Note that in some browsers it does not work, then you can stop the
interactive plot by clicking the blue on/off button in the upper right
corner, and then save the plot as any other figure in your browser
(probably by left-clicking it).

Sometimes, you need larger fonts in a plot that you want to include in a
report.  Below is the same plot, but with larger fonts.  We also overrule
the placement of the legend.

.. literalinclude:: mpl-demo.py
   :start-after: f1
   :end-at: f2

.. image:: f2.svg

More advanced example with multiple sub-plots.

.. literalinclude:: mpl-demo.py
   :start-after: f2
   :end-at: f3

.. image:: f3.svg

Plotting a countour

.. literalinclude:: mpl-demo.py
   :start-after: f3

.. image:: f4.svg


===================================
ASE (atomic simulation environment)
===================================

More details can be found here: https://ase-lib.org/index.html


Everything starts with a structure!
===================================

In ASE the most important ingredient is :class:`~ase.Atoms`
object used to setup an atomic structure.


Setting op a molecule using the ``Atoms`` object
------------------------------------------------

>>> from ase import Atoms
>>> d = 1.1
>>> co = Atoms('CO', positions=[[0, 0, 0], [0, 0, d]])

>>> # lets try and visualize it using the build in viewer in ase
>>> from ase.visualize import view
>>> view(co)


Setting up a periodic structure
-------------------------------

>>> d = 2.9
>>> L = 10
>>> wire = Atoms('Au', positions=[[0, L / 2, L / 2]],
...              cell=[d, L, L], pbc=[1, 0, 0])
>>> # lets try and repeat it and visualize primitive and repeated
>>> wire10 = wire * (10, 1, 1)
>>> view([wire, wire10])


Nitrogen on copper
==================

Exercise of the relaxation of a molecule on a surface
-----------------------------------------------------

This section gives a quick (and incomplete) overview of what ASE can do.

We will calculate the adsorption energy of a nitrogen molecule on a copper
surface. This is done by calculating the total energy for the isolated slab
and for the isolated molecule. The adsorbate is then added to the slab and
relaxed, and the total energy for this composite system is calculated. The
adsorption energy is obtained as the sum of the isolated energies minus the
energy of the composite system.

You can read more about the optimizers in ASE here:
https://ase-lib.org/ase/optimize.html

1. Try to go through the script so you understand what is going on

2. Calculate the adsorption energy of N2 on a 4x4x2 fcc111 slab
   (result= 0.324 eV)

3. Try a couple of different optimizers and see which one is the fastest

.. literalinclude:: n2cu111.py

.. code::

    # Visualize the trajectory
    $ ase gui N2Cu.traj


Band structure
==============

Using ASE to setup band structures for Al using a Freelectron model and DFT

1. What is the crystal structure of Al?

2. Try and look up the recommeded Brillouin zone path for crystal structure
   `here <https://ase-lib.org/ase/dft/kpoints.html>`__

3. Can you figure out what the ``nbands=-10`` and
   ``convergence={'bands': -5}`` parameters means in the GPAW DFT
   input below ? (Hint try and look at the output file ``Al.txt``)

.. literalinclude:: al.py
   :end-before: gpaw

Setup a DFT calculation with GPAW and repeat

.. literalinclude:: al.py
   :start-after: plot
