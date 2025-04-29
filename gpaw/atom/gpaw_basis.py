from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Sequence
from operator import methodcaller
from typing import Any, NamedTuple

from gpaw.atom.basis import BasisMaker
from gpaw.basis_data import Basis, parse_basis_name
from gpaw.typing import Self


class BasisInfo(NamedTuple):
    zetacount: int
    polcount: int
    name: str | None = None

    @classmethod
    def from_name(cls, name: str) -> Self:
        zc, pc = parse_basis_name(name)
        return cls(zc, pc, name)


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
    add_arguments(add)
    add('--save-setup', action='store_true',
        help='save setup to file')
    return parser


def parse_j_values(j: str) -> list[int]:
    return [int(value) for value in j.split(',')]


def parse_tail_norm(tail: str) -> list[float]:
    return [float(value) for value in tail.split(',')]


def add_arguments(add: Callable) -> None:
    add('-t', '--type',
        default='dzp', metavar='<type>', type=BasisInfo.from_name,
        help='type of basis.  For example: sz, dzp, qztp, '
        '4z3p.  [default: %(default)s]')
    add('-E', '--energy-shift',
        default=.1, metavar='<energy>', type=float,
        help='use given energy shift to determine cutoff')
    add('-T', '--tail-norm',
        default=[0.16, 0.3, 0.6], dest='tailnorm',
        metavar='<norm>[,<norm>[,...]]', type=parse_tail_norm,
        help='use the given fractions to define the split'
        '-valence cutoffs.  Default: [%(default)s]')
    add('--rcut-max',
        default=16., metavar='<rcut>', type=float,
        help='max cutoff for confined atomic orbitals.  '
        'This option has no effect on orbitals with smaller cutoff '
        '[default/Bohr: %(default)s]')
    add('--rcut-pol-rel', default=1.0, metavar='<rcut>', type=float,
        help='polarization function cutoff relative to largest '
        'single-zeta cutoff [default: %(default)s]')
    add('--rchar-pol-rel', metavar='<rchar>', type=float,
        help='characteristic radius of Gaussian when not using interpolation '
        'scheme, relative to rcut')
    add('--vconf-amplitude', default=12., metavar='<alpha>', type=float,
        help='set proportionality constant of smooth '
        'confinement potential [default: %(default)s]')
    add('--vconf-rstart-rel', default=.6, metavar='<ri/rc>', type=float,
        help='set inner cutoff for smooth confinement potential '
        'relative to hard cutoff [default: %(default)s]')
    add('--vconf-sharp-confinement', action='store_true',
        help='use sharp rather than smooth confinement potential')
    add('--lpol', metavar='<l>', type=int,
        help='angular momentum quantum number of polarization function.  '
        'Default behaviour is to take the lowest l which is not '
        'among the valence states')
    add('--jvalues', metavar='<j>[,<j>[,...]]', type=parse_j_values,
        help='explicitly specify which states to include.  '
        'Numbering corresponds to generator\'s valence state ordering.  '
        'For example: 0,1,2')


bad_density_warning = """\
Bad initial electron density guess!  Try rerunning the basis generator
with the '-g' parameter to run a separate non-scalar relativistic
all-electron calculation and use its resulting density as an initial
guess."""

very_bad_density_warning = """\
Could not generate non-scalar relativistic electron density guess,
or non-scalar relativistic guess was not good enough for the scalar
relativistic calculation.  You probably have to use the Python interface
to the basis generator in gpaw.atom.basis directly and choose very
smart parameters."""


def get_basis_maker_caller(args: Namespace) -> Callable[[BasisMaker], Basis]:
    if args.vconf_sharp_confinement:
        vconf_args = None
    else:
        vconf_args = args.vconf_amplitude, args.vconf_rstart_rel
    return methodcaller('generate', args.type.zetacount, args.type.polcount,
                        tailnorm=args.tailnorm,
                        energysplit=args.energy_shift,
                        rcutpol_rel=args.rcut_pol_rel,
                        rcutmax=args.rcut_max,
                        rcharpol_rel=args.rchar_pol_rel,
                        vconf_args=vconf_args,
                        l_pol=args.lpol,
                        jvalues=args.jvalues)


def main(args: Sequence[str] | None = None) -> None:
    from gpaw.atom.basisfromfile import read_setupdata

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
