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
