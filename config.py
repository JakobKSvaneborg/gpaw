# Copyright (C) 2006 CSC-Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.
import os
import sys
import re
import shlex
from sysconfig import get_platform
from subprocess import run
from pathlib import Path
from stat import ST_MTIME


def mtime(path, name, mtimes):
    """Return modification time.

    The modification time of a source file is returned.  If one of its
    dependencies is newer, the mtime of that file is returned.
    This function fails if two include files with the same name
    are present in different directories."""

    include = re.compile(r'^#\s*include "(\S+)"', re.MULTILINE)

    if name in mtimes:
        return mtimes[name]
    t = os.stat(os.path.join(path, name))[ST_MTIME]
    for name2 in include.findall(open(os.path.join(path, name)).read()):
        path2, name22 = os.path.split(name2)
        if name22 != name:
            t = max(t, mtime(os.path.join(path, path2), name22, mtimes))
    mtimes[name] = t
    return t


def check_dependencies(sources):
    # Distutils does not do deep dependencies correctly.  We take care of
    # that here so that "python setup.py build_ext" always does the right
    # thing!
    mtimes = {}  # modification times

    # Remove object files if any dependencies have changed:
    plat = get_platform() + '-{maj}.{min}'.format(maj=sys.version_info[0],
                                                  min=sys.version_info[1])
    remove = False
    for source in sources:
        path, name = os.path.split(source)
        t = mtime(path + '/', name, mtimes)
        o = 'build/temp.%s/%s.o' % (plat, source[:-2])  # object file
        if os.path.exists(o) and t > os.stat(o)[ST_MTIME]:
            print('removing', o)
            os.remove(o)
            remove = True

    so = 'build/lib.{}/_gpaw.so'.format(plat)
    if os.path.exists(so) and remove:
        # Remove shared object C-extension:
        # print 'removing', so
        os.remove(so)


def write_configuration(define_macros, include_dirs, libraries, library_dirs,
                        extra_link_args, extra_compile_args,
                        runtime_library_dirs, extra_objects, compiler):

    # Write the compilation configuration into a file
    try:
        out = open('configuration.log', 'w')
    except IOError as x:
        print(x)
        return
    print("Current configuration", file=out)
    print("compiler", compiler, file=out)
    print("libraries", libraries, file=out)
    print("library_dirs", library_dirs, file=out)
    print("include_dirs", include_dirs, file=out)
    print("define_macros", define_macros, file=out)
    print("extra_link_args", extra_link_args, file=out)
    print("extra_compile_args", extra_compile_args, file=out)
    print("runtime_library_dirs", runtime_library_dirs, file=out)
    print("extra_objects", extra_objects, file=out)
    out.close()


def build_gpu(gpu_compiler, gpu_compile_args, gpu_include_dirs,
              define_macros, undef_macros, build_temp):
    print("building gpu code", flush=True)

    kernels_dpath = Path('c/gpu/kernels')

    def create_build_dir(build_temp_base: Path, path_to_code: Path) -> None:
        """Creates a temp build directory corresponding to given directory in
        the code folder structure"""
        path_to_create = build_temp_base / path_to_code
        if not path_to_create.exists():
            print(f'creating {path_to_create}', flush=True)
            path_to_create.mkdir(parents=True)

    # Create temp build directory for gpu/kernels
    create_build_dir(build_temp, kernels_dpath)

    # Glob all kernel files, but remove those included by other kernels
    kernels = sorted(kernels_dpath.glob('*.cpp'))
    for name in ['interpolate-stencil.cpp',
                 'lfc-reduce.cpp',
                 'lfc-reduce-kernel.cpp',
                 'reduce.cpp',
                 'reduce-kernel.cpp',
                 'restrict-stencil.cpp']:
        kernels.remove(kernels_dpath / name)

    # Add other C++ code
    cpp_dpath = Path("c/gpu/cpp")
    create_build_dir(build_temp, cpp_dpath)
    cpp_files = sorted(cpp_dpath.glob("*.cpp"))

    cpp_files.extend(kernels)

    ## Add Magma-specific files if needed (figure out from define_macros)
    with_magma: bool = any(t[0] == "GPAW_WITH_MAGMA" for t in define_macros)
    if with_magma:
        magma_dpath = Path(cpp_dpath / "magma")
        create_build_dir(build_temp, magma_dpath)

        files_magma = sorted(magma_dpath.glob("*.cpp"))
        cpp_files.extend(files_magma)

    # Compile C++ code with cuda/hip compiler
    objects = []
    for src in cpp_files:
        obj = build_temp / src.with_suffix('.o')
        objects.append(str(obj))

        run_args = [gpu_compiler]
        run_args += gpu_compile_args

        for (name, value) in define_macros:
            arg = f'-D{name}'
            if value is not None:
                arg += f'={value}'
            run_args += [arg]
        run_args += [f'-U{name}' for name in undef_macros]
        run_args += [f'-I{dpath}' for dpath in gpu_include_dirs]
        run_args += ['-c', str(src)]
        run_args += ['-o', str(obj)]
        print(shlex.join(run_args), flush=True)
        p = run(run_args, check=False, shell=False)
        if p.returncode != 0:
            print(f'error: command {repr(gpu_compiler)} failed '
                  f'with exit code {p.returncode}',
                  file=sys.stderr, flush=True)
            sys.exit(1)

    return objects
