from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Sequence

from ..basis_data import Basis
from ..setup_data import SetupData
from .basis import BasisMaker
from .gpaw_basis import (
    add_arguments as _add_args, get_basis_maker_caller, read_setupdata)


def add_arguments(parser) -> None:
    add = parser.add_argument
    add('file', metavar='<filename>', nargs='+', help='setup data file')
    add('--name',
        metavar='<name>', help='basis name to be included in output filename')
    _add_args(add)


def generate_basis(setupdata: SetupData,
                   caller: Callable[[BasisMaker], Basis],
                   tokens: Sequence[str]):
    from gpaw.atom.all_electron import ValenceData

    valdata = ValenceData.from_setupdata_onthefly_potentials(setupdata)
    basis = caller(BasisMaker(valdata))

    # Should the setupname be added as part of the name, too?
    # Probably not, since we don't include the xcname either.
    # But I suppose it depends more on the runtime behaviour when
    # GPAW actually picks setups/basis sets for a calculation.
    outputfile = '.'.join([setupdata.symbol, *tokens])

    with open(outputfile, 'w') as fd:
        basis.write_to(fd)


def main(args: Namespace) -> None:
    caller = get_basis_maker_caller(args)
    tokens = []
    if args.name:
        tokens.append(args.name)
    tokens += [args.type.name, 'basis']
    for filename in args.file:
        print(f'Generating basis set for {filename!r}')
        setupdata = read_setupdata(filename)
        generate_basis(setupdata, caller, tokens)


class CLICommand:
    """Create basis sets from setup files."""

    add_arguments = staticmethod(add_arguments)
    run = staticmethod(main)
