# siteconfig.py file for MacOS with Apple Silicon processors, using homebrew.

fftw = True
scalapack = True
libraries = ['xc', 'openblas', 'fftw3', 'scalapack']

library_dirs = ['/opt/homebrew/lib']
include_dirs = ['/opt/homebrew/include']

# OpenBLAS
library_dirs += ['/opt/homebrew/opt/openblas/lib']
include_dirs += ['/opt/homebrew/opt/openblas/include']

# FFTW
library_dirs += ['/opt/homebrew/opt/fftw/lib']
include_dirs += ['/opt/homebrew/opt/fftw/include']

# Scalapack
library_dirs += ['/opt/homebrew/opt/scalapack/lib']

# Libxc
library_dirs += ['/opt/homebrew/opt/libxc/lib']
include_dirs += ['/opt/homebrew/opt/libxc/include']
