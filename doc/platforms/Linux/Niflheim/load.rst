.. _load on niflheim:

======================================
Using a pre-installed GPAW at Niflheim
======================================

This is the guide for using the pre-installed GPAW modules on Niflheim.

Modules on Niflheim
===================

You can see which modules are available with the ``module avail [package]`` command, for example::


  $ module avail GPAW

  -------------------------- /home/modules/modules/all --------------------------
     GPAW-setups/0.9.20000
     GPAW-setups/24.1.0
     GPAW-setups/24.11.0                            (D)
     GPAW/21.6.0-foss-2020b-ASE-3.22.0
     GPAW/21.6.0-foss-2020b-libxc-5.1.5-ASE-3.22.0
     GPAW/21.6.0-intel-2020b-ASE-3.22.0
     GPAW/21.6.0-intel-2020b-libxc-5.1.5-ASE-3.22.0
     GPAW/22.8.0-foss-2020b-ASE-3.22.1
     GPAW/22.8.0-foss-2020b-libxc-5.1.5-ASE-3.22.1
     GPAW/22.8.0-foss-2022a
     GPAW/22.8.0-intel-2020b-ASE-3.22.1
     GPAW/22.8.0-intel-2020b-libxc-5.1.5-ASE-3.22.1
     GPAW/22.8.0-intel-2022a
     GPAW/23.9.1-foss-2023a
     GPAW/23.9.1-intel-2023a
     GPAW/24.1.0-foss-2022a
     GPAW/24.1.0-foss-2023a
     GPAW/24.1.0-intel-2022a
     GPAW/24.1.0-intel-2023a
     GPAW/24.6.0-foss-2023a-ASE-3.23.0
     GPAW/24.6.0-intel-2023a-ASE-3.23.0
     GPAW/25.1.0-foss-2023a-ASE-3.24.0
     GPAW/25.1.0-foss-2023a-ASE-3.25.0
     GPAW/25.1.0-intel-2023a-ASE-3.24.0
     GPAW/25.1.0-intel-2023a-ASE-3.25.0
     GPAW/25.7.0-foss-2025b-CUDA-12.9.1
     GPAW/25.7.0-foss-2025b
     GPAW/25.7.0-intel-2025b                        (D)
     gpaw-data/1.0.1-GCCcore-14.3.0                 (D)
    Where:
     D:  Default Module

You can see which modules you have loaded with ``module list``.  You
can unload all modules to start from scratch with ``module purge``.


Choose the right version of GPAW
================================

This is a brief guide to which version of GPAW you should use. It
reflects the situation in December 2020 and will be updated as
the situation changes.


I have an ongoing project
  You should probably continue to use the version you are using in
  that project, unless you want to change.  See the section below on
  using different versions for different project.

I am a normal user
  You should load ``GPAW/25.7.0-intel-2025``.

  This will give the newest version of GPAW, as recommended by the
  developers.  It has new features and is significantly faster, in
  particular on the new Xeon40 nodes.  For ongoing projects that have
  been using an older version, you may find that some values have
  changed slightly - check for consistency, or be sure to always use
  the same version for ongoing projects.  See below for a description
  on how to do that.

I am sligtly conservative or need ``libvwdxc``.
  The version of GPAW compiled with the FOSS toolchain (``GPAW/25.7.0-foss-2025b``) is somewhat
  slower in many situations, but is better tested and may use less
  memory.  You may also have to use this version if you want the
  functionality from ``libvwdxc`` library, but be aware that many vad
  der Waals potentials do not use ``libvwdxc``.
  

**IMPORTANT:**  You do *not* need to load Python, ASE, matplotlib etc.
Loading GPAW pulls all that stuff in, in versions consistent with the
chosen GPAW version.

If you want to generate Wannier functions with the Wannier90 module,
you need to explicitly load ``Wannier90/3.1.0-foss-2025b`` or
``Wannier90/3.1.0-intel-2025b``.


Intel or foss versions?
=======================

The versions built with the Intel compilers and the Intel Math Kernel
Library (MKL) are in average faster than the ones build with the Open
Source (GNU) compilers (FOSS = Free and Open Source Software).  On
newer hardware this difference can be very significant, and we
recommend using the Intel versions unless you have a good reason not
to.

The ``libvdwcx`` library of van der Waals exchange-correlation
potentials in incompatible with the MKL, so if you need these methods
you have to use the foss versions.  However, most van der Waals
calculations use the native van der Waals support in GPAW, and works
fine with the Intel versions.



Module consistency is important: check it.
==========================================

For a reliable computational experience, you need to make sure that
all modules come from the same toolchain (i.e. that the software is
compiled with a consistent set of tools).  Some of the older
toolchains are only available on older Niflheim nodes.

**All modules you load should belong to the same toolchain.**

Use ``module list`` to list your modules. Check for consistency:

==================  ==================================
Toolchain           Module suffixes
==================  ==================================
foss/2025b          foss-2025b

                    gfbf-2025b

                    gompi-2025b

                    GCC-14.3.0

                    GCCcore-14.3.0
------------------  ----------------------------------
intel/2025b         intel-2025b

                    iimkl-2025b

                    iimpi-2025b

                    intel-compilers-2025.2.0

                    GCCcore-14.3.0
------------------  ----------------------------------
foss/2023a          foss-2023a

                    gfbf-2023a

                    gompi-2023a

                    GCC-12.3.0

                    GCCcore-12.3.0
------------------  ----------------------------------
intel/2023a         intel-2023a

                    iimkl-2023a

                    iimpi-2023a

                    intel-compilers-2023.1.0

                    GCCcore-12.3.0
------------------  ----------------------------------
foss/2022a          foss-2022a

                    gompi-2022a

                    GCC-11.3.0

                    GCCcore-11.3.0
------------------  ----------------------------------
intel/2022a         intel-2022a

                    iimpi-2022a

                    intel-compilers-2022.1.0

                    GCCcore-11.3.0
------------------  ----------------------------------
foss/2020b          foss-2020b

                    gompi-2020b

                    GCC-10.2.0

                    GCCcore-10.2.0
------------------  ----------------------------------
intel/2020b         intel-2020b

                    iimpi-2020b

                    iccifort-2020.4.304

                    GCCcore-10.2.0
------------------  ----------------------------------
fosscuda-2020b (*)  fosscuda-2020b

                    gompic-2020b

                    gcccuda-2020b

                    GCC-10.2.0

                    GCCcore-10.2.0
==================  ==================================

(*) For use on the GPU nodes, so only available on the sm3090 and xeon40 
partitions.  Newer toolchains do not use a special toolchain for this.

If your ``module load XXX`` commands give warnings about reloaded
modules, you are almost certainly mixing incompatible toolchains.


Using different versions for different projects.
================================================

You do not have to use the same modules for all your projects.  If you
want all jobs submitted from the folder ``~/ProjectAlpha`` to run with
an one version of GPAW, but everything else with a another version,
you can put this in your .bashrc::

  if [[ $SLURM_SUBMIT_DIR/ = $HOME/ProjectAlpha* ]]; then
      # Extreme consistency is important for this old project
      module purge
      module load GPAW/21.6.0-foss-2020b-ASE-3.22.0
  else
      # Performance is important for everything else.
      module purge
      module load GPAW/25.7.0-intel-2025b
      module load scikit-learn/1.7.1-iimkl-2025b
  fi

The ``module purge`` command is because SLURM will remember which
modules you have loaded when you submit the job, which must then be
unloaded.
