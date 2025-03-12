def read_setupdata(path):
    from pathlib import Path
    from gpaw.setup_data import SetupData
    path = Path(path)
    tokens = path.name.split('.')

    # We should not get symbol and xc from the filename, instead we should
    # parse them.
    #
    # Also, this doesn't work if the setup is unnamed.

    setupdata = SetupData(symbol=None, xcsetupname=None, readxml=False)
    setupdata.read_xml(source=path.read_bytes())
    return setupdata


def build_parser(parser):
    add = parser.add_argument
    add('file', nargs='+')


def generate(args, filename):
    from gpaw.atom.basis import BasisMaker

    setupdata = read_setupdata(filename)
    print(setupdata)

    # bm = BasisMaker()

def main(args):
    for filename in args.file:
        generate(args, filename)


class CLICommand:
    """Create basis sets from setup files."""

    @staticmethod
    def add_arguments(parser):
        build_parser(parser)

    @staticmethod
    def run(args):
        main(args)
