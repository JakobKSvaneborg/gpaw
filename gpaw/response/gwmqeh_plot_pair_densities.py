"""Plot pair densities and their mQEH-basis LS fits.

Standalone, laptop-friendly diagnostic. Reads a dump file written by
``GWmQEHCorrection(... dump_pair_densities=True)`` (the file lives next
to the regular mQEH output as ``<filename>_mqeh_pair_densities.npz``)
and plots, for each saved (rho_mz, drho, d) tuple:

  * the DFT in-plane pair density rho_mz(z), already in the qeh frame
    (i.e. evaluated at z_qeh + z_offset, so its peak should sit at the
    target layer's z_qeh_layer)
  * the rho-LS reconstruction rho_mqeh(z) = sum_a d_a rho_a(z)
  * a vertical line at z_qeh_layer

A good fit means the basis can represent the pair density;
visually-different curves are the smoking-gun for a frame mismatch,
an under-sized basis, or aliased periodic images leaking into the
projection.

Usage
-----

In your GPAW run, switch on the dump:

    gwq = GWmQEHCorrection(
        ...,
        filename='mono_gwmqeh',
        dump_pair_densities=True,          # writes
        dump_pair_densities_max_samples=40,  # first N samples
    )

Then on a laptop:

    python -m gpaw.response.gwmqeh_plot_pair_densities \\
        mono_gwmqeh_mqeh_pair_densities.npz \\
        --out pair_densities.pdf

Requires only numpy and matplotlib (NOT gpaw or qeh).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_dump(path):
    data = np.load(path, allow_pickle=False)
    return {
        'z_z_qeh':       data['z_z_qeh'],
        'dz_qeh':        float(data['dz_qeh']),
        'z_qeh_layer':   float(data['z_qeh_layer']),
        'rho_mz':        data['rho_mz'],       # (n, nz)
        'drho_za':       data['drho_za'],      # (n, nz, nb)
        'd_a':           data['d_a'],          # (n, nb)
        'iq':            data['iq'],           # (n,)
        'ig':            data['ig'],           # (n,)
        'm':             data['m'],            # (n,)
        'qplusGpar_abs': data['qplusGpar_abs'],  # (n,)
    }


def residual(rho, rho_fit):
    """||rho - rho_fit||_2 / ||rho||_2 (per-sample). Uses pixel L2."""
    norm = np.linalg.norm(rho)
    if norm < 1e-30:
        return 0.0
    return float(np.linalg.norm(rho - rho_fit) / norm)


def make_plot(dump, out_path, max_panels=12, separate_complex=True,
              show_basis=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = dump['z_z_qeh']
    z_layer = dump['z_qeh_layer']
    rho_all = dump['rho_mz']
    drho_all = dump['drho_za']
    d_all = dump['d_a']
    iq_all = dump['iq']
    ig_all = dump['ig']
    m_all = dump['m']
    qabs_all = dump['qplusGpar_abs']

    n = min(len(rho_all), max_panels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             sharex=True, squeeze=False)

    for k in range(n):
        ax = axes[k // ncols][k % ncols]
        rho = rho_all[k]
        drho = drho_all[k]                          # (nz, nb)
        d = d_all[k]
        rho_fit = drho @ d                          # (nz,)
        res = residual(rho, rho_fit)

        if separate_complex:
            ax.plot(z, rho.real, color='C0', lw=1.6, label='Re rho_mz')
            ax.plot(z, rho.imag, color='C0', lw=1.0, ls='--',
                    label='Im rho_mz')
            ax.plot(z, rho_fit.real, color='C3', lw=1.4,
                    label='Re fit')
            ax.plot(z, rho_fit.imag, color='C3', lw=0.9, ls='--',
                    label='Im fit')
        else:
            ax.plot(z, np.abs(rho), color='C0', lw=1.6, label='|rho_mz|')
            ax.plot(z, np.abs(rho_fit), color='C3', lw=1.4,
                    label='|fit|')

        if show_basis:
            for a in range(drho.shape[1]):
                ax.plot(z, drho[:, a].real, color='gray', lw=0.6,
                        alpha=0.5)

        ax.axvline(z_layer, color='black', lw=0.8, ls=':',
                   label='z_qeh_layer' if k == 0 else None)
        ax.set_title(
            f'iq={iq_all[k]} ig={ig_all[k]} m={m_all[k]}  '
            f'|q+G_par|={qabs_all[k]:.3f} Bohr$^{{-1}}$\n'
            f'residual = {res:.3f}',
            fontsize=10)
        ax.set_xlabel('z (Bohr)')
        ax.set_ylabel('density (Bohr$^{-1}$)')
        if k == 0:
            ax.legend(loc='best', fontsize=8)

    # Hide unused panels
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle(
        f'mQEH pair-density LS fits — {n} samples — z_qeh_layer = '
        f'{z_layer:.3f} Bohr',
        fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=150)
    print(f'wrote {out_path}')


def summary(dump):
    z = dump['z_z_qeh']
    z_layer = dump['z_qeh_layer']
    n_samples = len(dump['rho_mz'])
    print(f'samples              : {n_samples}')
    print(f'z_qeh_layer          : {z_layer:.4f} Bohr')
    print(f'qeh z-grid           : [{z[0]:.3f}, {z[-1]:.3f}] Bohr, '
          f'dz = {dump["dz_qeh"]:.4f} Bohr ({len(z)} points)')

    residuals = []
    for k in range(n_samples):
        rho = dump['rho_mz'][k]
        drho = dump['drho_za'][k]
        d = dump['d_a'][k]
        rho_fit = drho @ d
        residuals.append(residual(rho, rho_fit))
    residuals = np.array(residuals)
    print(f'residual stats       : '
          f'min={residuals.min():.4f}, '
          f'median={np.median(residuals):.4f}, '
          f'max={residuals.max():.4f}, '
          f'mean={residuals.mean():.4f}')
    high = residuals > 0.25
    print(f'samples with res>0.25: {int(high.sum())} / {n_samples}')


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split('\n\n', 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('npz_file', type=Path,
                   help='Path to <filename>_mqeh_pair_densities.npz')
    p.add_argument('--out', '-o', type=Path, default=None,
                   help='Output plot file (PDF or PNG). '
                        'Defaults to <npz_file>.pdf')
    p.add_argument('--max-panels', type=int, default=12,
                   help='Maximum number of (m, G_par) samples to plot '
                        '(default: 12)')
    p.add_argument('--abs', action='store_true',
                   help='Plot |rho| and |fit| instead of '
                        'Re/Im separately')
    p.add_argument('--show-basis', action='store_true',
                   help='Overlay the rho_a basis functions in gray')
    p.add_argument('--summary-only', action='store_true',
                   help='Just print a text summary and exit')
    args = p.parse_args(argv)

    dump = load_dump(args.npz_file)
    summary(dump)
    if args.summary_only:
        return 0

    out = args.out or args.npz_file.with_suffix('.pdf')
    make_plot(
        dump,
        out,
        max_panels=args.max_panels,
        separate_complex=not args.abs,
        show_basis=args.show_basis,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
