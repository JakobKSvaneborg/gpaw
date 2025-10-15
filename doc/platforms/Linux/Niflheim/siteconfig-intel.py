import os

scalapack = True
intelmkl = True
fftw = True

if os.getenv('EBVERSIONINTEL') >= '2025':
    mpicompiler = 'mpiicx'
else:
    mpicompiler = 'mpiicc'

# Use Intel MKL
libraries = ['xc', 'mkl_sequential', 'mkl_core', 'fftw3xc_intel_pic', 'mkl_rt']
mkl = os.getenv('MKLROOT')
include_dirs = [os.path.join(mkl, 'include')]

# Use EasyBuild scalapack from the active toolchain
libraries += ['mkl_scalapack_lp64', 'mkl_blacs_intelmpi_lp64']

# Use EasyBuild libxc
libxc = os.getenv('EBROOTLIBXC')
include_dirs += [os.path.join(libxc, 'include')]

# libvdwxc:
# Use EasyBuild libvdwxc
# This will only work with the foss toolchain.
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
    library_dirs = [os.path.join(elpa, 'lib')]
    extra_link_args = [f'-Wl,-rpath={elpa}/lib']
    include_dirs.append(os.path.join(elpa, 'include', 'elpa-' + elpaversion))

# Now add a EasyBuild "cover-all-bases" library_dirs
library_dirs += os.getenv('LD_LIBRARY_PATH').split(':')
