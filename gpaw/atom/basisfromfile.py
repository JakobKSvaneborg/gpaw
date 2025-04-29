from pathlib import Path


def read_setupdata(path):
    from gpaw.setup_data import SetupData
    path = Path(path)

    setupdata = SetupData(symbol=None, xcsetupname=None, readxml=False)
    setupdata.read_xml(source=path.read_bytes())
    return setupdata


def build_parser(parser):
    add = parser.add_argument
    add('file', nargs='+')
    add('--name', help='Basis name to be included in output filename.')
    add('--type', default='dzp',
        help='Type of basis set.  Currently only "dzp".')


def generate_basis(setupdata, args):
    from gpaw.atom.basis import BasisMaker
    from gpaw.atom.all_electron import ValenceData

    valdata = ValenceData.from_setupdata_onthefly_potentials(setupdata)
    bm = BasisMaker(valdata)
    basis = bm.generate()

    # Should the setupname be added as part of the name, too?
    # Probably not, since we don't include the xcname either.
    # But I suppose it depends more on the runtime behaviour when
    # GPAW actually picks setups/basis sets for a calculation.
    tokens = [setupdata.symbol]
    if args.name:
        tokens.append(args.name)
    tokens += [args.type, 'basis']
    outputfile = '.'.join(tokens)

    with open(outputfile, 'w') as fd:
        basis.write_to(fd)


def main(args):
    for filename in args.file:
        print(f'Generating basis set for {filename}')
        setupdata = read_setupdata(filename)
        generate_basis(setupdata, args)


class CLICommand:
    """Create basis sets from setup files."""

    @staticmethod
    def add_arguments(parser):
        build_parser(parser)

    @staticmethod
    def run(args):
        main(args)
