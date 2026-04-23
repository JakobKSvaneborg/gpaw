"""Convert an LCAO gpw file to PW mode for response/BSE calculations.

Requires new GPAW (run with ``GPAW_NEW=1`` in the environment, or import
via ``from gpaw.new.ase_interface import GPAW``).
"""
import argparse
import os
import sys

from gpaw import GPAW


def convert(src, dst):
    dft = GPAW(src).dft
    mode = dft.ibzwfs.mode
    if mode != 'lcao':
        print(f'{src}: mode is {mode!r}, nothing to do.')
        return

    dft.change(eigensolver={})
    dft.change_mode('pw')
    ecut_eV = dft.params.mode.ecut
    print(f'Converted {src} -> {dst}')
    print(f'  PW ecut = {ecut_eV:.3f} eV (default)')
    dft.write_gpw_file(dst, include_wfs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src', help='input LCAO gpw file')
    parser.add_argument('dst', help='output PW gpw file')
    args = parser.parse_args()

    if not os.environ.get('GPAW_NEW'):
        print('Warning: GPAW_NEW is not set; this script requires new GPAW.',
              file=sys.stderr)
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
