def read_setupdata(path):
    from pathlib import Path
    from gpaw.setup_data import SetupData
    path = Path(path)
    tokens = path.name.split('.')

    # We should not get symbol and xc from the filename, instead we should
    # parse them.
    #
    # Also, this doesn't work if the setup is unnamed.
    symbol = tokens[0]
    xc = tokens[2]

    setupdata = SetupData(symbol, xc, readxml=False)
    setupdata.read_xml(source=path.read_bytes())
    return setupdata


def build_parser(parser):
    add = parser.add_argument
    add('file', nargs='+')


def main(args):
    print(args)


class CLICommand:
    """Create basis sets from setup files."""

    @staticmethod
    def add_arguments(parser):
        build_parser(parser)

    @staticmethod
    def run(args):
        main(args)
