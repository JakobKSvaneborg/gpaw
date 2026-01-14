#!/usr/bin/env python
# Copyright (C) 2003-2020  CAMP
# Please see the accompanying LICENSE file for further information.
from __future__ import annotations

import functools
import os
import re
import runpy
import shlex
import sys
import tempfile
import textwrap
import traceback
import warnings
from pathlib import Path
from subprocess import run
from sysconfig import get_config_var, get_platform
from typing import Any, Callable

from distutils.ccompiler import new_compiler, CCompiler
from distutils.errors import CCompilerError
from distutils.sysconfig import customize_compiler
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

config = runpy.run_path(Path(__file__).parent / 'config.py')

build_gpu = config['build_gpu']
check_dependencies = config['check_dependencies']
write_configuration = config['write_configuration']


def warn_deprecated(msg):
    msg = f'\n\n{msg}\n\n'
    warnings.warn(msg, DeprecationWarning)


def raise_error(msg):
    msg = f'\n\n{msg}\n\n'
    raise ValueError(msg)


def config_args(key):
    return shlex.split(get_config_var(key))


# Deprecation check
for i, arg in enumerate(sys.argv):
    if arg.startswith('--customize='):
        custom = arg.split('=')[1]
        raise DeprecationWarning(
            f'Please set GPAW_CONFIG={custom} or place {custom} in ' +
            '~/.gpaw/siteconfig.py')

# temp flag for choosing different options for C/C++ builds.
# Currently this must be set to true in siteconfig.py if compiling as C++
use_cpp = False

libraries = ['xc']
library_dirs = []
include_dirs = []
extra_link_args = []
extra_compile_args = ['-Wall', '-Wno-unknown-pragmas']
runtime_library_dirs = []
extra_objects = []
define_macros = [('NPY_NO_DEPRECATED_API', '7'),
                 ('GPAW_NO_UNDERSCORE_CBLACS', None),
                 ('GPAW_NO_UNDERSCORE_CSCALAPACK', None),
                 ('GPAW_MPI_INPLACE', None)]
undef_macros = ['NDEBUG']

gpu_target = None
gpu_compiler = None
gpu_compile_args = []
gpu_include_dirs = []

PLACEHOLDER = object()
parallel_python_interpreter = PLACEHOLDER
compiler = None
mpi = None
fftw = False
scalapack = False
libvdwxc = False
elpa = False
gpu = False
magma = False
intelmkl = False

# If these are not defined, we try to resolve default values for them
noblas = None
nolibxc = None

# Advanced:
# If these are defined, they replace
# all the default args from setuptools
compiler_args = None
linker_so_args = None
linker_exe_args = None


def ensure_cpp_standard(compile_args: list):
    """GPAW C++ code requires -std=c++17. This adds it to the input compile
    args list if it's missing. Don't override existing flags, but warn if the
    user-specified standard is too low.
    """
    old_std = None
    for arg in compile_args:
        m = re.match(r'-std=(c\+\+(\d+))', arg)
        if m:
            old_std = m.group(1)
            version = int(m.group(2))
            if version < 17:
                warnings.warn(f"C++ standard {old_std} is too low, "
                              "GPAW requires -std=c++17 or newer")
            break

    if old_std is None:
        compile_args.append('-std=c++17')


# Search and store current git hash if possible
try:
    from ase.utils import search_current_git_hash
    githash = search_current_git_hash('gpaw')
    if githash is not None:
        define_macros += [('GPAW_GITHASH', githash)]
    else:
        print('.git directory not found. GPAW git hash not written.')
except ImportError:
    print('ASE not found. GPAW git hash not written.')

# User provided customizations:
gpaw_config = os.environ.get('GPAW_CONFIG')
if gpaw_config and not Path(gpaw_config).is_file():
    raise FileNotFoundError(gpaw_config)
for siteconfig in [gpaw_config,
                   'siteconfig.py',
                   '~/.gpaw/siteconfig.py']:
    if siteconfig is not None:
        path = Path(siteconfig).expanduser()
        if path.is_file():
            print('Reading configuration from', path)
            exec(path.read_text())
            break
else:  # no break
    if not noblas:
        libraries.append('blas')

if use_cpp:
    print("EXPERIMENTAL: Compiling entire GPAW as C++.")
    ensure_cpp_standard(extra_compile_args)

# Deprecation check
if 'mpicompiler' in locals():
    mpicompiler = locals()['mpicompiler']
    msg = 'Please remove deprecated declaration of mpicompiler.'
    if mpicompiler is None:
        mpi = False
        msg += (' Define instead in siteconfig one of the following lines:'
                '\n\nmpi = False\nmpi = True')
    else:
        mpi = True
        compiler = mpicompiler
        msg += (' Define instead in siteconfig:'
                f'\n\nmpi = True\ncompiler = {repr(compiler)}')
    warn_deprecated(msg)


# If `mpi` was not set in siteconfig,
# it is enabled by default if `mpicc` is found
default_mpi_compiler = 'mpic++' if use_cpp else 'mpicc'
if mpi is None:
    if compiler is None:
        if (os.name != 'nt'
                and run(['which', default_mpi_compiler],
                        capture_output=True).returncode == 0):
            mpi = True
            compiler = default_mpi_compiler
        else:
            mpi = False
    elif compiler in ['mpicc', 'mpic++', 'mpiCC', 'mpicxx']:
        warn_deprecated(
            'Define in siteconfig explicitly'
            '\n\nmpi = True')
        mpi = True
    else:
        mpi = False

if mpi:
    if compiler is None:
        raise_error('Define compiler for MPI in siteconfig:'
                    "\ncompiler = ...  # MPI compiler, e.g., 'mpicc'")


# Deprecation check
if 'mpilinker' in locals():
    mpilinker = locals()['mpilinker']
    msg = ('Please remove deprecated declaration of mpilinker:'
           f'\ncompiler={repr(compiler)} will be used for linking.')
    if mpilinker == compiler:
        warn_deprecated(msg)
    else:
        msg += ('\nPlease contact GPAW developers if you need '
                'different commands for linking and compiling.')
        raise_error(msg)

# Deprecation check
for key in ['libraries', 'library_dirs', 'include_dirs',
            'runtime_library_dirs', 'define_macros']:
    mpi_key = 'mpi_' + key
    if mpi_key in locals():
        warn_deprecated(
            f'Please remove deprecated declaration of {mpi_key}'
            f' and use only {key} instead.'
            f'\nAdding {mpi_key} to {key}.')
        locals()[key] += locals()[mpi_key]


if parallel_python_interpreter is not PLACEHOLDER:
    raise RuntimeError(
        'The "parallel_python_interpreter" keyword has been removed '
        'and the "gpaw-python" interpreter is no longer compiled.  '
        'Please modify your siteconfig.py accordingly.')


if mpi:
    print('Building GPAW with MPI support.')


if gpu:
    valid_gpu_targets = ['cuda', 'hip-amd', 'hip-cuda']
    if gpu_target not in valid_gpu_targets:
        raise ValueError('Invalid gpu_target in configuration: '
                         'gpu_target should be one of '
                         f'{str(valid_gpu_targets)}.')
    if gpu_compiler is None:
        if gpu_target.startswith('hip'):
            gpu_compiler = 'hipcc'
        elif gpu_target == 'cuda':
            gpu_compiler = 'nvcc'

    if gpu_target == 'cuda':
        gpu_compile_args += ['-x', 'cu']

    if '-fPIC' not in ' '.join(gpu_compile_args):
        if gpu_target in ['cuda', 'hip-cuda']:
            gpu_compile_args += ['-Xcompiler']
        gpu_compile_args += ['-fPIC']

    ensure_cpp_standard(gpu_compile_args)

    # GPU code needs to link to c++ stdlib. This is automatic if the linking
    # is done using a C++ compiler, if not we have to add it manually
    if not use_cpp:
        libraries += ['stdc++']


def set_compiler_executables(cc: CCompiler) -> None:
    # Override the compiler executables
    # A hack to change the used compiler and linker, inspired by
    # https://shwina.github.io/custom-compiler-linker-extensions/
    for (name, my_args) in [('compiler', compiler_args),
                            ('compiler_so', compiler_args),
                            ('compiler_cxx', compiler_args),
                            ('compiler_so_cxx', compiler_args),
                            ('linker_so', linker_so_args),
                            ('linker_exe', linker_exe_args),
                            ('linker_so_cxx', linker_so_args),
                            ('linker_exe_cxx', linker_exe_args)]:
        new_args = []
        old_args = getattr(cc, name)

        # Set executable
        if compiler is not None:
            new_args += [compiler]
        else:
            new_args += [old_args[0]]
        # Set args
        if my_args is not None:
            new_args += my_args
        else:
            new_args += old_args[1:]
        cc.set_executable(name, new_args)


def get_compiler() -> CCompiler:
    compiler = new_compiler()
    customize_compiler(compiler)
    set_compiler_executables(compiler)
    for name, dirs in [  # Convert Path objects to str
        ('library_dirs', library_dirs),
        ('include_dirs', include_dirs),
        ('runtime_library_dirs', runtime_library_dirs),
    ]:
        dirs = [str(dname) for dname in dirs]
        getattr(compiler, 'set_' + name)(dirs)
    compiler.set_libraries(libraries)
    for macro, value in define_macros:
        if macro in undef_macros:
            continue
        compiler.define_macro(macro, value)
    for macro in undef_macros:
        compiler.undefine_macro(macro)
    return compiler


def try_compiling(
    func: Callable[..., Any],
    catch_exceptions: (
        type[Exception] | tuple[type[Exception], ...]) = CCompilerError,
) -> Callable[[], bool]:
    @functools.wraps(func)
    def wrapper():
        try:
            c_program = textwrap.dedent(func.__doc__)
            compiler = get_compiler()
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = os.path.abspath(tmpdir)
                c_file_path = os.path.join(tmpdir, 'test.c')
                c_out_path = os.path.join(tmpdir, 'a.out')
                with open(c_file_path, mode='w') as fobj:
                    print(c_program, file=fobj)
                obj_files = compiler.compile([c_file_path],
                                             output_dir=tmpdir,
                                             extra_preargs=extra_compile_args)
                compiler.link_executable(obj_files,
                                         c_out_path,
                                         extra_preargs=extra_link_args)
        except catch_exceptions as e:
            traceback.print_exception(type(e), e, e.__traceback__,
                                      file=sys.stderr)
            return False
        else:
            return True

    return wrapper


@try_compiling
def test_libxc() -> bool:
    """
    #include "xc.h"

    void no_op(int _) { return; }  // Suppress -Wunused-variable

    int main(int argc, char *argv[]) {
        // Compiler would fail if `XC_MAJOR_VERSION` isn't defined
        no_op(XC_MAJOR_VERSION);
        return 0;
    }
    """


@try_compiling
def test_blas() -> bool:
    """
    // GPAW calls Fortran BLAS functions directly instead of using cblas.h,
    // so must forward declare the function here and prevent name mangling.

    # ifdef GPAW_NO_UNDERSCORE_BLAS
    #  define dnrm2_  dnrm2
    # endif

    #ifdef __cplusplus
    extern "C" {
    #endif

    double dnrm2_(int n, double *vec, int stride);
    #ifdef __cplusplus
    }
    #endif

    void no_op(double _) { return; }  // Suppress -Wunused-variable

    int main(int argc, char *argv[]) {
        // Linker would fail if `dnrm2_()` isn't found
        double vec[2] = { 3., 4. };
        no_op(dnrm2_(2, vec, 1));
    }
    """


class MissingLibraryWarning(UserWarning):
    pass


# Automated checks: resolve defaults for `noblas` and `nolibxc` where
# appropriate

if noblas is None:
    noblas = not test_blas()
    if noblas:
        warnings.warn('cannot link against BLAS (see above error), '
                      'setting `noblas = True`',
                      MissingLibraryWarning)
if nolibxc is None:
    nolibxc = not test_libxc()
    if nolibxc:
        warnings.warn('cannot link against LibXC (see above error), '
                      'setting `nolibxc = True`',
                      MissingLibraryWarning)

for flag, name in [(noblas, 'GPAW_WITHOUT_BLAS'),
                   (nolibxc, 'GPAW_WITHOUT_LIBXC'),
                   (mpi, 'PARALLEL'),
                   (fftw, 'GPAW_WITH_FFTW'),
                   (scalapack, 'GPAW_WITH_SL'),
                   (libvdwxc, 'GPAW_WITH_LIBVDWXC'),
                   (elpa, 'GPAW_WITH_ELPA'),
                   (intelmkl, 'GPAW_WITH_INTEL_MKL'),
                   (magma, 'GPAW_WITH_MAGMA'),
                   (gpu, 'GPAW_GPU'),
                   (gpu, 'GPAW_GPU_AWARE_MPI'),
                   (gpu and gpu_target == 'cuda',
                       'GPAW_CUDA'),
                   (gpu and gpu_target.startswith('hip'),
                       'GPAW_HIP'),
                   (gpu and gpu_target == 'hip-amd',
                       '__HIP_PLATFORM_AMD__'),
                   (gpu and gpu_target == 'hip-cuda',
                       '__HIP_PLATFORM_NVIDIA__'),
                   (use_cpp, 'GPAW_CPP'),
                   ]:
    if flag and name not in [n for (n, _) in define_macros]:
        define_macros.append((name, None))

sources = [Path('c/bmgs/bmgs.c')]
sources += Path('c').glob('*.c')
if gpu:
    sources += Path('c/gpu').glob('*.c')
sources += Path('c/xc').glob('*.c')

if nolibxc:  # Cleanup: remove stale refrerences to LibXC
    for name in ['libxc.c', 'm06l.c',
                 'tpss.c', 'revtpss.c', 'revtpss_c_pbe.c',
                 'xc_mgga.c']:
        sources.remove(Path(f'c/xc/{name}'))
    # Dynamic link
    try:
        libraries.remove('xc')
    except ValueError:
        pass
    # Static link
    for libxc in [
        static_lib for static_lib in extra_link_args
        if os.path.basename(static_lib) == 'libxc.a'
    ]:
        extra_link_args.remove(libxc)
if noblas:  # Cleanup: remove stale refrerences to BLAS
    try:  # Dynamic link
        libraries.remove('blas')
    except ValueError:
        pass

# Make build process deterministic (for "reproducible build")
sources = [str(source) for source in sources]
sources.sort()

check_dependencies(sources)

# Convert Path objects to str:
runtime_library_dirs = [str(dir) for dir in runtime_library_dirs]
library_dirs = [str(dir) for dir in library_dirs]
include_dirs = [str(dir) for dir in include_dirs]

define_macros = [(macro, value) for macro, value in define_macros
                 if macro not in undef_macros]

extensions = [Extension('_gpaw',
                        sources,
                        libraries=libraries,
                        library_dirs=library_dirs,
                        include_dirs=include_dirs,
                        define_macros=define_macros,
                        undef_macros=undef_macros,
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                        runtime_library_dirs=runtime_library_dirs,
                        extra_objects=extra_objects,
                        language='c++' if use_cpp else 'c')]


write_configuration(define_macros, include_dirs, libraries, library_dirs,
                    extra_link_args, extra_compile_args,
                    runtime_library_dirs, extra_objects, compiler)


class build_ext(_build_ext):

    def run(self):
        import numpy as np
        self.include_dirs.append(np.get_include())

        if self.link_objects is None:
            self.link_objects = []

        if gpu:
            objects = build_gpu(gpu_compiler, gpu_compile_args,
                                gpu_include_dirs + self.include_dirs,
                                define_macros, undef_macros,
                                self.build_temp)

            self.link_objects += objects

        super().run()

    def build_extensions(self):
        set_compiler_executables(self.compiler)
        super().build_extensions()
        print("Build temp:", self.build_temp)
        print("Build lib: ", self.build_lib)


data = 'git+https://gitlab.com/gpaw/gpaw-web-page-data.git'
setup(ext_modules=extensions, cmdclass={'build_ext': build_ext})
