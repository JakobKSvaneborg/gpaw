mpi = True
compiler = 'mpicc'
libraries = []
library_dirs = []
include_dirs = []
extra_compile_args = [
    '-O3',
    '-march=native',
    '-mtune=native',
    '-mavx2',
    '-fopenmp',  # implies -fopenmp-simd
]
extra_link_args = ['-fopenmp']

# blas
libraries += ['openblas']

# scalapack
scalapack = True
libraries += ['scalapack']
# fftw
fftw = True
libraries += ['fftw3']

# libxc
libraries += ['xc']

# libvdwxc
libvdwxc = True
libraries += ['vdwxc']

define_macros += [('GPAW_ASYNC', 1)]
