import argparse


description = """\
GPAW benchmark suite. Provides a list of unchanging benchmarks which allow \
to track speed and memory usage of GPAW calculations over time.\
"""
run_benchmark_help = """\
Run a list of benchmarks. Alternatively, users can use special filter \
commands to select benchmarks containing particular style of calculations, \
and thus easily create subsets of benchmarks.\
"""

benchmarks_help = """\
Run the given list of benchmarks and produce benchmarking data. \
Benchmarks can be either nicknames. or long description strings. \
Run `python -m gpaw.benchmark list` for extended list of standard \
benchmark nicknames, and the explanation of how to create custom \
benchmarks.\
"""

list_benchmark_help = """\
Give a list of all benchmark nicknames, and their respective long names, \
and a brief description of the systems and the parameter sets utilized in \
the long names. Users can then use this information to build their own \
benchmark calculation parameter sets for particular systems.\
"""

version = "May 2025"

if __name__ == '__main__':
    from gpaw.benchmark import benchmark_main
    parser = argparse.ArgumentParser(prog='gpaw.benchmark',
                                     description=description)
    subparsers = parser.add_subparsers(help='subcommand help', dest='command')
    run_parser = subparsers.add_parser('run', help=run_benchmark_help)
    run_parser.add_argument('benchmarks', nargs='*', help=benchmarks_help)
    list_parser = subparsers.add_parser('list', help=list_benchmark_help)

    args = parser.parse_args()
    if args.command == 'run':
        for benchmark in args.benchmarks:
            benchmark_main(benchmark)
    else:
        raise ValueError(f'Invalid command {args.command}.')
