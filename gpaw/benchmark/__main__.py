import argparse


description = """\
GPAW benchmark suite. Provides a list of unchanging benchmarks, we can track optimizations with.
"""

version = "May 2025"

if __name__ == '__main__':
    from gpaw.benchmark import benchmark_main
    parser = argparse.ArgumentParser(prog='gpaw.benchmark',
                                     description=description)
    parser.add_argument('benchmark')
    parser.add_argument('--version', action='version',
                        version=version)
    args = parser.parse_args()
    benchmark_main(args.benchmark)

