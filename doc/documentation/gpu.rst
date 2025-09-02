.. _gpu:

GPU
===

Ground-state calculations on a GPU is a new feature with
some limitations:

* only PW-mode
* it has only been implemented in the new GPAW code

You use the new code like this:

>>> from gpaw.new.ase_interface import GPAW
>>> atoms = ...
>>> atoms.calc = GPAW(..., parallel={'gpu': True})

By default, the environment variable ``$GPAW_USE_GPUS`` is used, to determine
whether to use gpu or not (defaults to not).
In addition, the user can specify ``parallel={‘gpu’: False}`` (or True) to
override this behaviour.

Instead of importing ``GPAW`` from ``gpaw.new.ase_interface``, you can use ``from gpaw import GPAW`` and the select new GPAW
by setting the environment variable :envvar:`GPAW_NEW` to ``1``:
``GPAW_NEW=1 python ...``.
See :git:`gpaw/test/gpu/test_pw.py` for an example.

The GPAW CI has a GitLab Runner with a GPU, so the GPU parts of GPAW are tested by the GPAW's test suite as well.

.. envvar:: GPAW_NEW

   If this environment variable is set to ``1`` then new GPAW will be used, when it is imported as
   ``from gpaw import GPAW``. The other method to use the new GPAW, which does not require the environment variable
   is to import it from ``gpaw.new.ase_interface``.

.. envvar:: GPAW_USE_GPUS

   If this environment variable is set to ``1`` then the default value for ``gpu`` in the parallel
   dictionary will be set to ``True``. Since it only is a default value,
   the effect of ``$GPAW_USE_GPUS`` may be overrided by specifying
   the ``gpu`` key to the ``parallel`` dictionary.

.. envvar:: GPAW_CPUPY

   If this environment variable is set to ``1``, then users without GPU's can run the GPU code.
   CuPy will be emulated by fictious library cpupy. This option is useful to make sure
   that developers without GPU do not break the GPU code.


.. tip::

   >>> import numpy as np
   >>> from gpaw.gpu import cupy as cp
   >>> a_cpu = np.zeros(...)
   >>> a_gpu = cp.asarray(a_cpu)  # from CPU to GPU
   >>> b_cpu = a_gpu.get()  # from GPU to CPU

Building the GPU code
---------------------

To build GPAW with GPU support, siteconfig.py needs to be updated. To see how to use siteconfig, see :ref:`siteconfig`. Five variables need to be set:

    1. ``gpu`` is a boolean determining whether to build the GPU kernels or not.
    2. ``gpu_target`` where valid target architectures are ``'cuda'``, ``'hip-amd'`` or ``'hip-cuda'``. Essentially, with NVIDIA architectures, the target should be ``'cuda'``, and ``nvcc``  compiler will be required, and with ``hip-`` selections, ``hipcc`` compiler will be used.
    3. ``gpu_compiler`` is optional, and will be selected by ``gpu_target`` normally, but it can be overwritten with this parameter.
    4. ``gpu_include_dirs`` are not normally needed, but can be used to provide additional search paths to locate headers.
    5. ``gpu_compile_args`` is essential, and proper target architecture needs to be supplied in most cases.


In addition, libraries list should be appended by GPU blas and GPU runtime librarires. See the examples below for examples of how to utilize these commands.

Example piece of siteconfig to build with HIP (AMD MI250X)::

    gpu = True
    gpu_target = 'hip-amd'
    gpu_compiler = 'hipcc'
    gpu_include_dirs = []
    gpu_compile_args = [
        '-g',
        '-O3',
        '--offload-arch=gfx90a',
        ]
    libraries += ['amdhip64', 'hipblas']

Example piece of siteconfig to build with CUDA (NVIDIA A100)::

    gpu = True
    gpu_target = 'cuda'
    gpu_compiler = 'nvcc'
    gpu_compile_args = ['-O3',
                        '-g',
                        '-gencode', 'arch=compute_80,code=sm_80']

    libraries += ['cudart', 'cublas']


To see what the siteconfig should look in practice, see
:download:`../platforms/Cray/siteconfig-lumi-gpu.py`
(AMD MI250X) or
:download:`../platforms/Linux/Niflheim/siteconfig-foss.py`
(NVIDIA A100) examples.


GPU parallelization
-------------------

Same parallelization options are available as with the CPU version.
GPAW will distribute the available GPUs in round robin manner.
As a rule of thumb, always use 1 CPU per logical GPU. While it rarely helps to oversubscribe the GPUs, it might sometimes give a small speed up.

By default, GPAW will utilize GPU-aware MPI, expecting the MPI library to be compiled with GPU-aware MPI support.
However, if this is not the case (segfaults or bus errors occur at MPI calls),
one may disable the GPU-aware MPI with following commmand added to the siteconfig::

    undef_macros += ['GPAW_GPU_AWARE_MPI']

If disabled, at MPI calls, GPAW will transfer data from GPU to CPU, to move it via MPI in CPU, and transfer it back to GPU after that. However, the normal behaviour is to tranfer directly from GPU to GPU.

The gpaw.gpu module
===================

.. module:: gpaw.gpu

.. data:: cupy

   :mod:`cupy` module (or :mod:`gpaw.gpu.cpupy` if :mod:`cupy` is not available)

.. data:: cupyx

   ``cupyx`` module (or :mod:`gpaw.gpu.cpupyx` if ``cupyx`` is not available)

.. autodata:: cupy_is_fake
.. autodata:: is_hip
.. autofunction:: as_np
.. autofunction:: as_xp
.. autofunction:: cupy_eigh


Fake cupy library
=================

.. module:: gpaw.gpu.cpupy
.. module:: gpaw.gpu.cpupyx

The implementation uses cupy_.  In the code, we don't do ``import cupy as cp``.
Instead we use ``from gpaw.gpu import cupy as cp``.  This allows us to use a
fake ``cupy`` implementation so that we can run GPAW's ``cupy`` code without
having a physical GPU.  To enable the fake ``cupy`` module, do::

  GPAW_CPUPY=1 python ...

This allows users without a GPU to find out if their code interferes with the
GPU implementation, simply by running the tests.

.. _cupy: https://cupy.dev/


CuPy enabled container objects
==============================

The following objects:

* :class:`~gpaw.core.UGArray`
* :class:`~gpaw.core.PWArray`
* :class:`~gpaw.core.atom_arrays.AtomArrays`
* :class:`~gpaw.core.matrix.Matrix`

can have their data (``.data`` attribute) stored in a :class:`cupy.ndarray`
array instead of, as normal, a :class:`numpy.ndarray` array.  In additions,
these objects now have an ``xp`` attribute that can be :mod:`numpy` or
:mod:`cupy`.

Also, the :class:`~gpaw.core.atom_centered_functions.AtomCenteredFunctions`
object can do its operations on the GPU.


Building GPAW with MAGMA support
================================

.. _MAGMA: https://icl.utk.edu/magma/

GPAW provides wrappers to a subset of eigensystem solvers from the MAGMA_
library, which implements efficient, hybrid CPU-GPU algorithms for common linear
algebra tasks. Compiling GPAW with MAGMA support is recommended for performance
if running on AMD GPUs. On Nvidia there is currently no performance increase.

MAGMA features can be enabled in siteconfig.py::

   magma = True
   libraries += ['magma']

You may also need to modify ``library_dirs``, ``runtime_library_dirs`` and
``include_dirs`` with paths to your MAGMA installation (see :ref:`siteconfig`).

You will also need to ensure the CUDA/HIP compiler standard is set to C++17 or newer (``-std=c++17``).
Modern CUDA/HIP installations do this automatically, and GPAW installation also adds this flag.
In case you still face issues:
   1. If your ``siteconfig.py`` adds ``'-std=...''`` to ``gpu_compile_args``, update the standard there.
   GPAW will not override a user-defined standard.
   2. If using HIP to compile CUDA code (`hipcc` as a wrapper to `nvcc`), you may need to set the standard through an environment variable:
   ``export HIPCC_COMPILE_FLAGS_APPEND="-std=c++17"``.
   However, we generally recommend using `nvcc` and the CUDA toolkit directly if building for Nvidia GPUs.

You can use the ``gpaw.cgpaw.have_magma`` flag to check if MAGMA is available
within your GPAW installation. GPAW eigensystem routines will default to the MAGMA implementation
on AMD GPUs, provided the matrix is large enough to benefit from it. You can
also call the MAGMA solvers directly from the ``gpaw.new.magma`` module.
