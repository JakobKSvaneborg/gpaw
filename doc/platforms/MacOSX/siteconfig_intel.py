# siteconfig.py file for MacOS with Intel processors, using homebrew.

fftw = True
scalapack = True
libraries = ['xc', 'openblas', 'fftw3', 'scalapack']

library_dirs = ['/usr/local/lib']
include_dirs = ['/usr/local/include']

# OpenBLAS
library_dirs += ['/usr/local/opt/openblas/lib']
include_dirs += ['/usr/local/opt/openblas/include']

# FFTW
library_dirs += ['/usr/local/opt/fftw/lib']
include_dirs += ['/usr/local/opt/fftw/include']

# Scalapack
library_dirs += ['/usr/local/opt/scalapack/lib']

# libxc
library_dirs += ['/usr/local/opt/libxc/lib']
include_dirs += ['/usr/local/opt/libxc/include']

