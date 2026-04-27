import os

#compiler = 'gcc'
mpicompiler = 'mpicc'
scalapack = True
fftw = True

libraries = ['mkl_intel_lp64',
            'mkl_sequential',
            'mkl_lapack',
             'mkl_core',
             'pthread',
             'readline',
             'termcap',
             'xc',
             'mkl_blacs_intelmpi_lp64']

# FFTW should be configured from environment variables, but they do
# not report the correct names for a dynamically loaded library.
# Use Intel MKL
if fftw:
    libraries += ['fftw3xc_intel_pic', 'mkl_rt']

# Use EasyBuild scalapack from the active toolchain
if scalapack:
    libraries += ['mkl_scalapack_lp64','mkl_lapack95_lp64']

# Use EasyBuild libxc
libxc = os.getenv('EBROOTLIBXC')
include_dirs.append(os.path.join(libxc, 'include'))

# libvdwxc:
# Use EasyBuild libvdwxc
# NOTE: This currenlty does not work together with the Intel MKL, so
# the easyconfig files does not load libvdwxc
libvdwxc = os.getenv('EBROOTLIBVDWXC')
if libvdwxc:
    include_dirs.append(os.path.join(libvdwxc, 'include'))
    libraries.append('vdwxc')

# ELPA:
# Use EasyBuild ELPA if loaded
elpa = os.getenv('EBROOTELPA')
if elpa:
    libraries += ['elpa']
    elpaversion = os.path.basename(elpa).split('-')[0]
    library_dirs.append(os.path.join(elpa, 'lib'))
    extra_link_args += ['-Wl,-rpath={}/lib'.format(elpa)]
    include_dirs.append(os.path.join(elpa, 'include', 'elpa-'+elpaversion))

# Now add a EasyBuild "cover-all-bases" library_dirs
library_dirs = os.getenv('LD_LIBRARY_PATH').split(':')
