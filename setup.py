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
import subprocess
from sysconfig import get_config_var, get_platform
from typing import Any, Callable, Optional

from distutils.ccompiler import new_compiler, CCompiler
from distutils.errors import CCompilerError
from distutils.sysconfig import customize_compiler
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

config = runpy.run_path(Path(__file__).parent / 'config.py')

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

# Globals that can be replaced by values in user siteconfig.py

libraries = ['xc']
library_dirs = []
include_dirs = []
extra_link_args = []
extra_compile_args = ['-Wall', '-Wno-unknown-pragmas', '-std=c99']
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

makefile_build: bool = False
"""EXPERIMENTAL: Flag that enables generation of GPAW Makefiles and compiling
the C++ through make, instead of going through the usual setuptools build path.
This is intended for developers that actively modify the C++ code and has much
better support for incremental builds over setuptools.

If enabled, the following changes apply:
- A Makefile is produced in GPAW root folder that has the same compiler inputs
as what setuptools would use (see caveat below).
- Build directory for .o files is forcibly changed to _build instead of some
temporary location.
- Compiler flags are slightly reorganized, so that extra_compile_args get
passed BEFORE the input .cpp file instead of AFTER. The latter is what
setuptools likes to do for whatever reason.
- The _gpaw.*.so extension will be built with make. Setuptools does no
compilation.
- `extra_objects` is not supported.

Intended usage: Set `makefile_build = True` in siteconfig.py, make an editable
install with `pip install -v -e . --no-build-isolation`, then run `make`
whenever you change a source file. This will recompile only the sources that
have changed and rebuild the *.so module by running the link stage again.
Without `--no-build-isolation` the include path to Numpy's headers would point
to a temporary pip directory.

If changing siteconfig.py, you should reconstruct the Makefile by running the
pip step again, and probably run `make clean`.

See also the `configure_only` flag.
"""

configure_only: bool = False
"""EXPERIMENTAL: If True, will not compile anything. Use with `makefile_build`
to quickly re-generate the Makefile after changing siteconfig.py."""


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
if mpi is None:
    if compiler is None:
        if (os.name != 'nt'
                and subprocess.run(['which', 'mpicc'],
                                   capture_output=True).returncode == 0):
            mpi = True
            compiler = 'mpicc'
        else:
            mpi = False
    elif compiler == 'mpicc':
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

    # Some C++ code (eg. magma wrappers) requires C++17 standard.
    # Add the flag here, but don't override any user-specified option.
    # GPAW C++ files that require C++17 are encouraged to #error
    # with a sensible message if the standard is too low.
    has_std_flag = (
        any(re.match(r'-std=.+', arg) for arg in gpu_compile_args)
    )
    if not has_std_flag:
        print("Adding gpu compile argument: '-std=c++17'")
        gpu_compile_args += ['-std=c++17']

    # GPU code needs to link to c++ stdlib
    libraries += ['stdc++']


def set_compiler_executables(cc: CCompiler) -> None:
    # Override the compiler executables
    # A hack to change the used compiler and linker, inspired by
    # https://shwina.github.io/custom-compiler-linker-extensions/
    for (name, my_args) in [('compiler', compiler_args),
                            ('compiler_so', compiler_args),
                            ('linker_so', linker_so_args),
                            ('linker_exe', linker_exe_args)]:
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
    // For some reasons GPAW doesn't seem to use any of the BLAS header
    // files (e.g. 'cblas.h' and 'f77blas.h'), but instead declares
    // them explicitly (e.g. 'c/blas.c');
    // do as the Romans do
    #ifdef GPAW_NO_UNDERSCORE_BLAS
    #  define dnrm2_  dnrm2
    #endif

    double dnrm2_(int n, double *vec, int stride);

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
                        language='c')]


write_configuration(define_macros, include_dirs, libraries, library_dirs,
                    extra_link_args, extra_compile_args,
                    runtime_library_dirs, extra_objects, compiler)


def parse_cflags(define_macros: Optional[list[tuple[str, str]]],
                 undef_macros: Optional[list[str]],
                 include_dirs: Optional[list[str]]
                 ) -> list[str]:
    """Converts setuptools-style compiler args to a form that corresponds to
    CFLAGS in Makefiles. Example output:
        ['-DSOME_DEFINE', '-USOME_UNDEF', '-I/some/path']

    Note that flags like -O3, -fopenmp etc must be added separately.
    See the BuildGPAW class.
    """
    cflags = []

    if define_macros:
        for name, val in define_macros:
            if val is None:
                cflags.append(f"-D{name}")
            else:
                cflags.append(f"-D{name}={val}")
    if undef_macros:
        cflags += [f"-U{m}" for m in undef_macros]

    if include_dirs:
        cflags += [f"-I{d}" for d in include_dirs]

    return cflags


def parse_ldflags(libraries: Optional[list[str]],
                  library_dirs: Optional[list[str]],
                  runtime_library_dirs: Optional[list[str]]) -> list[str]:
    """Converts setuptools-style linker args to a form that corresponds to
    LDFLAGS in Makefiles. Example output:
        ['-lsomelib', '-L/some/lib/path/', '-Wl,-rpath,/some/rpath/']
    """
    ldflags = []

    if library_dirs:
        ldflags += [f"-L{d}" for d in library_dirs]
    if runtime_library_dirs:
        ldflags += [f"-Wl,-rpath,{rp}" for rp in runtime_library_dirs]
    if libraries:
        ldflags += [f"-l{lib}" for lib in libraries]

    return ldflags


def build_gpu(gpu_compiler: str,
              gpu_compile_args: list[str],
              gpu_include_dirs: list[str],
              define_macros: list[tuple[str, str]],
              undef_macros: list[str],
              build_dir: str,
              inout_makefile_lines: Optional[list[str]] = None):
    """Manually builds all CUDA/HIP source files with `gpu_compiler`.
    Return value is a list of objects that can be added as extra
    link_objects to setuptools when building the main GPAW extension.
    If `inout_makefile_lines` is given, will append a corresponding Makefile
    to this list.

    TODO: reduce the number of source files that need this special compilation
    path. Currently there is GPU code in common headers so many otherwise
    normal .cpp files end up becoming CUDA/HIP code.
    """
    info_msg = "Building gpu code" if not configure_only else "Configuring GPU build"
    print(info_msg, flush=True)

    def create_build_dir(path_to_code: Path) -> None:
        """Creates subdirectory under build_dir with same folder structure as
        that of the input source file"""
        path_to_create = build_dir / path_to_code
        if not path_to_create.exists():
            print(f'creating {path_to_create}', flush=True)
            path_to_create.mkdir(parents=True)

    # Create temp build directory for c/gpu/kernels
    kernels_dpath = Path('c/gpu/kernels')
    create_build_dir(kernels_dpath)

    # Glob all kernel files, but remove those #included by other kernels
    kernels = sorted(kernels_dpath.glob('*.cpp'))
    for name in ['interpolate-stencil.cpp',
                 'lfc-reduce.cpp',
                 'lfc-reduce-kernel.cpp',
                 'reduce.cpp',
                 'reduce-kernel.cpp',
                 'restrict-stencil.cpp']:
        kernels.remove(kernels_dpath / name)

    # Add other code that ends up using CUDA/HIP features
    cpp_dpath = Path("c/gpu/cpp")
    create_build_dir(cpp_dpath)
    cpp_files = sorted(cpp_dpath.glob("*.cpp"))

    cpp_files.extend(kernels)

    # Add Magma-specific files if needed
    if magma:
        magma_dpath = Path(cpp_dpath / "magma")
        create_build_dir(magma_dpath)

        files_magma = sorted(magma_dpath.glob("*.cpp"))
        cpp_files.extend(files_magma)

    # Combine flags, and pass the correct language flag.
    # TODO what to do if user has passed their own -x that is not compatible?
    cflags = gpu_compile_args.copy()
    if '-x' not in gpu_compile_args:
        lang = 'cu' if gpu_target == 'cuda' else 'hip'
        print(f"Adding GPU compilation flag: -x{lang}")
        cflags += [f'-x{lang}']

    cflags += parse_cflags(define_macros, undef_macros, gpu_include_dirs)

    should_write_makefile = (inout_makefile_lines is not None)
    if should_write_makefile:
        inout_makefile_lines.append("# BEGIN GPU SECTION\n")
        inout_makefile_lines.append(f"GPU_SOURCES := {" ".join([str(src) for src in cpp_files])}\n")
        inout_makefile_lines.append(f"GPU_BUILD_DIR := {build_dir}")
        inout_makefile_lines.append(f"GPU_OBJECTS := $(addprefix $(GPU_BUILD_DIR)/,$(addsuffix .o,$(basename $(GPU_SOURCES))))")
        inout_makefile_lines.append(f"GPU_DEPS := $(GPU_OBJECTS:.o=.d)")
        inout_makefile_lines.append(f"\nCC_GPU := {gpu_compiler}\n")
        cflags_str = " ".join(cflags)
        inout_makefile_lines.append(f"CFLAGS_GPU := {cflags_str} -MMD -MP\n")

    # Compile with cuda/hip compiler
    objects = []
    for src in cpp_files:
        obj = build_dir / src.with_suffix('.o')
        objects.append(str(obj))

        run_args = [gpu_compiler]
        run_args += cflags
        run_args += ['-c', str(src)]
        run_args += ['-o', str(obj)]

        if should_write_makefile:
            inout_makefile_lines.append(f"{obj}: {src}\n\t mkdir -p $(dir $@)\n\t$(CC_GPU) $(CFLAGS_GPU) -c {src} -o {obj}\n")

        if not configure_only:
            print(shlex.join(run_args), flush=True)
            p = subprocess.run(run_args, check=False, shell=False)

            if p.returncode != 0:
                print(f'error: command {repr(gpu_compiler)} failed '
                      f'with exit code {p.returncode}',
                      file=sys.stderr, flush=True)
                sys.exit(1)

    if should_write_makefile:
        inout_makefile_lines.append("-include $(GPU_DEPS)")
        inout_makefile_lines.append("# END GPU SECTION\n")

    return objects


class BuildGPAW(build_ext):
    """"""

    @property
    def makefile_build_dir(self) -> str:
        """Build dir to use when doing a Makefile based build.
        This is relative to the Makefile (project root).
        """
        return "_build"

    def run(self):
        """"""
        import numpy as np
        self.include_dirs.append(np.get_include())

        self.makefile_lines = []
        if makefile_build:
            # Set a persistent build directory. We get a more readable Makefile when using relative build paths instead
            self.build_temp = os.path.join(os.path.dirname(__file__), self.makefile_build_dir)
            os.makedirs(self.build_temp, exist_ok=True)

            # _gpaw.cpython-3XX-ARCH-PLATFORM.so
            module_names = [self.get_ext_filename(ext.name) for ext in self.extensions]
            module_names_str = " ".join(name for name in module_names)

            self.makefile_lines.append("# MAKEFILE GENERATED BY GPAW BUILD SYSTEM\n")

            self.makefile_lines.append("all: " + module_names_str)
            self.makefile_lines.append(f"\nclean:\n\trm -rf {self.makefile_build_dir} " + module_names_str + "\n")

        if self.link_objects is None:
            self.link_objects = []

        if gpu:
            assert gpu_compiler
            # NB: we don't add any default flags from setuptools here
            objects = build_gpu(gpu_compiler, gpu_compile_args,
                                gpu_include_dirs + self.include_dirs,
                                define_macros, undef_macros,
                                self.makefile_build_dir,
                                self.makefile_lines)

            self.link_objects += objects

        super().run()

    def build_extensions(self):
        """Called from super().run()"""

        set_compiler_executables(self.compiler)

        makefile_local = []
        if makefile_build:
            makefile_local.append(f"SOURCES := {" ".join([str(src) for src in sources])}\n")
            makefile_local.append(f"BUILD_DIR := {self.makefile_build_dir}\n")
            makefile_local.append("OBJECTS := $(addprefix $(BUILD_DIR)/,$(addsuffix .o,$(basename $(SOURCES))))")
            makefile_local.append("DEPS := $(OBJECTS:.o=.d)")
            makefile_local.append(f"\nCC := {self.compiler.compiler_so[0]}")

            for ext in self.extensions:

                # Object filenames. We get a more readable Makefile by writing them relative to Makefile location
                objs = [self.compiler.object_filenames([src], output_dir=self.makefile_build_dir)[0] for src in ext.sources]

                # Python and Numpy includes are added to self, NOT to the
                # extension. So take them, plus any user-specified includes.
                includes = self.include_dirs + ext.include_dirs

                # Compile flags. Combine "base" flags from setuptools and those
                # from extra_compile_args. NOTE! Ordering is slightly different
                # from setuptools method: Setuptools puts extra_compile_args
                # at the very end, after the input file. Here we add extras
                # immediately after "base" flags.

                # "base"
                cflags = list(self.compiler.compiler_so[1:])
                if ext.extra_compile_args:
                    cflags += ext.extra_compile_args
                # Build -D, -U, -I flags
                cflags += parse_cflags(ext.define_macros,
                                       ext.undef_macros,
                                       includes)

                # Linker flags
                ldflags = list(self.compiler.linker_so[1:])
                if ext.extra_link_args:
                    ldflags += ext.extra_link_args

                ldflags += parse_ldflags(ext.libraries + self.libraries,
                                         ext.library_dirs + self.library_dirs,
                                         ext.runtime_library_dirs)
                # setuptools adds this too:
                ldflags.insert(0, '-Wl,--enable-new-dtags')

                cflags_str = " ".join(cflags)
                ldflags_str = " ".join(ldflags)

                # Add CFLAGS and LDFLAGS, with extra flags for dependency generation
                makefile_local.append(f"\nCFLAGS := {cflags_str} -MMD -MP\n")
                makefile_local.append(f"LDFLAGS := {ldflags_str}\n")

                # Define build target. Need to include .o files from GPU part
                target_name = self.get_ext_filename(ext.name)
                target_objects = "$(OBJECTS)"
                if gpu:
                    # Put GPU objects first so that they are built first
                    target_objects = " $(GPU_OBJECTS) " + target_objects

                makefile_local.append(f"{target_name}: {target_objects}\n\t$(CC) {target_objects} -o $@ $(LDFLAGS)\n")

                # Compile rules
                for src, obj in zip(ext.sources, objs):
                    makefile_local.append(f"{obj}: {src}\n\t mkdir -p $(dir $@)\n\t$(CC) $(CFLAGS) -c {src} -o {obj}\n")

            makefile_local.append("\n-include $(DEPS)")
            self.makefile_lines += makefile_local

            with open("Makefile", "w") as mf:
                mf.write("\n".join(self.makefile_lines))
            print("Generated Makefile")

        if configure_only:
            print(
                "\n"
                "***********************************************************\n"
                "NOTE: your siteconfig is using `configure_only = True`.\n"
                "GPAW C-extension has NOT been built.\n"
                "You can either compile it manually using the generated\n"
                "Makefile, or removing the configure_only flag and\n"
                "rerunning this installation.\n"
                "***********************************************************\n"
                "\n")
            return

        if makefile_build:
            # run make
            print("Makefile build: running `make`", flush=True)
            p = subprocess.run(["make"], check=False, shell=False)

            if p.returncode != 0:
                print(f'error: command make failed '
                      f'with exit code {p.returncode}',
                      file=sys.stderr, flush=True)
                sys.exit(1)
        else:
            # Build normally with setuptools
            super().build_extensions()

        print("Build temp:", self.build_temp)
        print("Build lib: ", self.build_lib)

    def copy_extensions_to_source(self):
        """Override to prevent copy errors when building using `make`, which
        does not put the .so in a temp dir (self.build_lib). Also needed to
        prevent errors if using `configure_only = True`.
        """
        if not makefile_build:
            super().copy_extensions_to_source()


data = 'git+https://gitlab.com/gpaw/gpaw-web-page-data.git'
setup(ext_modules=extensions, cmdclass={'build_ext': BuildGPAW})
