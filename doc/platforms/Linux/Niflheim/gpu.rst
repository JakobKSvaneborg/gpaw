.. _gpu on niflheim:

============================================
Running GPU-GPAW with A100 nodes on Niflheim
============================================

To get started, you either need to

  * Load a GPU aware GPAW module, e.g. ``GPAW/25.7.0-foss-2025b-CUDA-12.9.1`` (only available on GPU nodes and the
    corresponding login nodes)

or

  * Build GPAW using the virtual environment script ``gpaw_venv.py``
    on this page :ref:`Compiling GPAW on Niflheim <build on
    niflheim>`.  This will automatically build the latest GPU version
    of GPAW.

Then follow this procedure:
   
1. Add A100 nodes to myqueue config.

   * Type ``mq info`` to locate the myqueue root, and the config.py file in
     there, and add following items to the list of nodes::

        ('a100G1', {'nodename': 'a100', 'cores': 128, 'memory': '512000M', 'extra_args':['--gpus-per-node=1', '--mem=100G']}),
        ('a100G2', {'nodename': 'a100', 'cores': 128, 'memory': '512000M', 'extra_args':['--gpus-per-node=2', '--mem=200G']}),
        ('a100G4', {'nodename': 'a100', 'cores': 128, 'memory': '512000M', 'extra_args':['--gpus-per-node=4', '--mem=400G']}),

   * This will allow you to run calculations with 1, 2 or 4 GPUs (4 GPUs equals a full node).


2. Use the latest ``gpaw_venv.py`` script to build the environment. This automatically will set ``GPAW_NEW=1``
   and ``GPAW_USE_GPUS=1`` environment variables when using a100 nodes. Thus, the user only has to submit to
   a100 nodes, and GPUs should work automatically. Check out the wave function memory section from the output, 
   it should have ``storage: GPU`` under it.
  
   If do not want to use the latest ``gpaw_venv.py``, alternatively, 
   you need to prepare your script for GPUs. One needs to use the NEW gpaw, when running with GPUs,
   so either you need to add ``export GPAW_NEW=1`` to your virtual environment,
   or import the calculator from new GPAW directly. In addition you need to add
   ``parallel={'gpu': True, ...}`` to your input of GPAW, or set ``export GPAW_USE_GPUS=1`` environment variable. 
   Here is an example script
   to relax a nanostructure (note that you can keep the absolute path, if you want to rerun this test)::

    def gpaw_test(atoms, gpu=False):
        kpts = {'density': 4}
        params = {'convergence': {'density': 1e-06},
                  'kpts': kpts,
                  'random': True,
                  'mode': {'ecut': 800, 'name': 'pw', 'force_complex_dtype': True},
                  'occupations': {'name': 'fermi-dirac', 'width': 0.05},
                  'mixer': {'method': 'fullspin', 'backend': 'pulay'},
                  'txt': f'relax_gpu_{gpu}.txt',
                  'parallel': {'gpu': gpu},
                  'xc': 'LDA'}

        from gpaw.new.ase_interface import GPAW
        atoms.calc = GPAW(**params)
        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        return E, F

    from ase.io import read
    atoms = read('/home/niflheim/kuisma/benchmarks/bilayer_example.json')
    import sys
    gpaw_test(atoms, gpu=eval(sys.argv[1]))

3. Submit with myqueue. We will submit two calculations, one with a full A100 node (4 GPUs), and one
   with the fastest CPU node (epyc96). Always select the number of cores equal to the number of total GPUs::

       mq submit -R 4:a100G4:1h 'gpaw python gpu_example.py True'
       mq submit -R 96:epyc96:1h 'gpaw python gpu_example.py False'

4. The system is a 256 atom bilayer. The expected runtime for this system is 5 minutes with A100 node, and 20 minutes with full epyc96 node::

     mq ls

        id      folder name args                           info res.           age state    time
      ─────── ────── ──── ────────────────────────────── ──── ──────────── ───── ─────── ─────
      7839868 ./     gpaw python gpu_example.py False    +3   96:epyc96:1h 21:05 done    19:41
      7839870 ./     gpaw python gpu_example.py True     +3   4:a100G4:1h  20:56 done     4:59


For reference, the runtime is 40 minutes with second fastest node Xeon56, so the current master version
of GPAW provides a factor of 4 to 8 node-to-node speedup, depending on the point of comparison. Note
that the XC-corrections are being performed on CPU a the moment in master, so the GPU code will
speed up further by 20-60% depending on the system, when certain merge requests are merged.

6. You may investigate the outputs of the calculations yourself, or you can observe the files already at Niflheim::

       sdiff /home/niflheim/kuisma/benchmarks/relax_gpu_True.txt /home/niflheim/kuisma/benchmarks/relax_gpu_False.txt|less


