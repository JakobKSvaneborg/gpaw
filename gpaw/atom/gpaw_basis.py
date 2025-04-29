from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from typing import Any

from .basis import BasisMaker
from .basisfromfile import (
    add_common_args, get_basis_maker_caller, read_setupdata)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Generate LCAO basis sets for '
                            'the specified elements.')
    add = parser.add_argument
    add('symbols', metavar='<symbol>', nargs='+', help='chemical symbols')
    add('--version', action='version', version='%(prog)s 0.1')
    add('-n', '--name', default=None, metavar='<name>',
        help='name of generated basis files')
    add('-f', '--xcfunctional', default='PBE', metavar='<XC>',
        help='exchange-Correlation functional [default: %(default)s]')
    add_common_args(add)
    add('--save-setup', action='store_true',
        help='save setup to file')
    return parser


def main(args: Sequence[str] | None = None) -> None:
    def generate_basis_set(symbol_or_path: str,
                           caller: Callable[[BasisMaker], Any], /,
                           **kwargs) -> None:
        if '.' in symbol_or_path:  # symbol is actually a path
            from gpaw.atom.all_electron import ValenceData
            setupdata = read_setupdata(symbol_or_path)
            valdata = ValenceData.from_setupdata_onthefly_potentials(setupdata)
            bm = BasisMaker(valdata, **kwargs)
        else:
            bm = BasisMaker.from_symbol(symbol_or_path, **kwargs)

        basis = caller(bm)
        basis.write_xml()

    parser = get_parser()
    arguments = parser.parse_args(args)
    caller = get_basis_maker_caller(arguments)

    for symbol in arguments.symbols:
        generate_basis_set(symbol, caller,
                           name=arguments.name,
                           xc=arguments.xcfunctional,
                           save_setup=arguments.save_setup)
