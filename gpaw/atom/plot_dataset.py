from __future__ import annotations

import textwrap
from ast import literal_eval
from types import SimpleNamespace
from xml.dom import minidom

from matplotlib import pyplot as plt

from ..basis_data import Basis, BasisPlotter
from ..setup_data import SetupData
from .aeatom import AllElectronAtom
from .generator2 import PAWSetupGenerator, generate, plot_log_derivs


def reconstruct_paw_gen(paw: str,
                        basis: str | None = None) -> PAWSetupGenerator:
    symbol, *remainder, setupname = paw.split('.')
    tag = '.'.join(remainder)
    setup = SetupData(symbol, setupname, readxml=False)
    with open(paw, mode='rb') as fobj:
        # `SetupData.vbar_g` can be read from the setup XML
        # (<zero_potential>)
        setup.read_xml(fobj.read())
    params = {'v0': None}
    generator_data = setup.generatordata
    if not generator_data:
        generator, = minidom.parse(paw).getElementsByTagName('generator')
        text, = generator.childNodes
        generator_data = textwrap.dedent(text.data).strip('\n')
    for line in generator_data.splitlines():
        key, _, value = line.rstrip(',').partition('=')
        try:
            value = literal_eval(value)
        except Exception:
            continue
        params[key] = value
    # XXX: Replace this with data read directly from the files
    gen = generate(**params)
    if False and basis is not None:
        name = 'dzp'
        if tag:
            name = f'{tag}.{name}'
        gen.basis = basis_obj = Basis(params['symbol'],
                                      name,
                                      readxml=False, rgc=gen.rgd)
        # `Basis.bf_j` and `.ribf_j` can be read from the basis XML
        # (<basis_function>)
        basis_obj.read_xml(basis)
    return gen


def main(args: SimpleNamespace,
         gen: PAWSetupGenerator | None = None,
         plot: bool = True) -> None:
    if args.create_basis_set in (True, False):
        basis_file = None
    else:
        basis_file = args.create_basis_set
        args.create_basis_set = True
    if gen is None:
        gen = reconstruct_paw_gen(args.paw, basis_file)
    if args.logarithmic_derivatives:
        plot_log_derivs(gen, args.logarithmic_derivatives, plot=True)
    if not plot:
        return
    gen.plot()
    if args.create_basis_set:
        if gen.basis is None:
            gen.create_basis_set()
        gen.basis.generatordata = ''  # we already printed this
        BasisPlotter(show=True).plot(gen.basis)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass


class CLICommand:
    """Plot PAW dataset."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('-b', '--basis-set',
            const=True,
            default=False,
            # For compatibility with `generator2`
            dest='create_basis_set',
            metavar='BASIS',
            nargs='?',
            help='Load the basis set from an XML file; '
            'if not provided, create a rudimentary basis set.')
        add('-l', '--logarithmic-derivatives',
            metavar='spdfg,e1:e2:de,radius',
            help='Plot logarithmic derivatives. ' +
            'Example: -l spdf,-1:1:0.05,1.3. ' +
            'Energy range and/or radius can be left out.')
        add('paw',
            metavar='DATASET',
            help='XML file from which to read the PAW dataset')

    @staticmethod
    def run(args):
        main(args)
