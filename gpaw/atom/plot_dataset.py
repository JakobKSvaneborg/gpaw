from __future__ import annotations

import textwrap
from ast import literal_eval
from collections.abc import Iterable
from types import SimpleNamespace
from xml.dom import minidom

from matplotlib import pyplot as plt

from .. import typing
from ..basis_data import Basis, BasisPlotter
from ..setup_data import SetupData
from .aeatom import colors
from .generator2 import PAWSetupGenerator, generate, plot_log_derivs
from .radialgd import AERadialGridDescriptor


_PartialWaveItem = tuple[int,  # l
                         int,  # n
                         float,  # r_cut
                         float,  # energy
                         typing.Array1D,  # phi_g
                         typing.Array1D]  # phit_g
_ProjectorItem = tuple[int,  # l
                       int,  # n
                       float,  # energy
                       typing.Array1D]  # pt_g


def plot_partial_waves(ax: plt.Axes,
                       rgd: AERadialGridDescriptor,
                       cutoff: float,
                       iterator: Iterable[_PartialWaveItem]) -> None:
    r_g = rgd.r_g
    i = 0
    for l, n, rcut, e, phi_g, phit_g in sorted(iterator):
        if n == -1:
            gc = rgd.ceil(rcut)
            name = '*{} ({:.2f} Ha)'.format('spdf'[l], e)
        else:
            gc = len(rgd)
            name = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
        ax.plot(r_g[:gc], (phi_g * r_g)[:gc], color=colors[i], label=name)
        ax.plot(r_g[:gc], (phit_g * r_g)[:gc], '--', color=colors[i])
        i += 1
    ax.axis(xmin=0, xmax=3 * cutoff)
    ax.set_xlabel('radius [Bohr]')
    ax.set_ylabel(r'partial waves $r\phi_{n\ell}(r)$')
    ax.legend()


def plot_projectors(ax: plt.Axes,
                    rgd: AERadialGridDescriptor,
                    cutoff: float,
                    iterator: Iterable[_ProjectorItem]) -> None:
    r_g = rgd.r_g
    i = 0
    for l, n, e, pt_g in sorted(iterator):
        if n == -1:
            name = '*{} ({:.2f} Ha)'.format('spdf'[l], e)
        else:
            name = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
        ax.plot(r_g, pt_g * r_g, color=colors[i], label=name)
        i += 1
    ax.axis(xmin=0, xmax=cutoff)
    ax.set_xlabel('radius [Bohr]')
    ax.set_ylabel(r'projectors $r\tilde{p}(r)$')
    ax.legend()


def get_ppw_params_paw_setup_generator(
    gen: PAWSetupGenerator,
) -> tuple[AERadialGridDescriptor, float, Iterable[_PartialWaveItem]]:
    return (gen.rgd,
            gen.rcmax,
            ((l, n, waves.rcut, e, phi_g, phit_g)
             for l, waves in enumerate(gen.waves_l)
             for n, e, phi_g, phit_g in zip(waves.n_n, waves.e_n,
                                            waves.phi_ng, waves.phit_ng)))


def get_ppw_params_setup_data(
    setup: SetupData,
) -> tuple[AERadialGridDescriptor, float, Iterable[_PartialWaveItem]]:
    return (setup.rgd,
            setup.r0,
            zip(setup.l_j, setup.n_j, setup.rcut_j, setup.eps_j,
                setup.phi_jg, setup.phit_jg))


def get_pp_params_paw_setup_generator(
    gen: PAWSetupGenerator,
) -> tuple[AERadialGridDescriptor, float, Iterable[_ProjectorItem]]:
    return (gen.rgd,
            gen.rcmax,
            ((l, n, e, pt_g)
             for l, waves in enumerate(gen.waves_l)
             for n, e, pt_g in zip(waves.n_n, waves.e_n, waves.pt_ng)))


def get_pp_params_setup_data(
    setup: SetupData,
) -> tuple[AERadialGridDescriptor, float, Iterable[_ProjectorItem]]:
    return (setup.rgd,
            setup.r0,
            zip(setup.l_j, setup.n_j, setup.eps_j, setup.pt_jg))


def reconstruct_paw_gen(paw: str,
                        basis: str | None = None) -> PAWSetupGenerator:
    setup = read_setup_file(paw)
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
        gen.basis = read_basis_file(basis)
    return gen


def read_basis_file(basis: str) -> Basis:
    symbol, *chunks, end = basis.split('.')
    assert end == 'basis'
    name = '.'.join(chunks)
    return Basis.read_xml(symbol, name, basis)


def read_setup_file(paw: str) -> SetupData:
    symbol, *_, setupname = paw.split('.')
    setup = SetupData(symbol, setupname, readxml=False)
    with open(paw, mode='rb') as fobj:
        # Can be read from the setup XML:
        # - `SetupData.vbar_g` (<zero_potential>)
        # - `SetupData.nc_g` (<ae_core_density>)
        # - `SetupData.nct_g` (<pseudo_core_density>)
        # - `SetupData.tauc_g` (<ae_core_kinetic_energy_density>)
        # - `SetupData.tauct_g` (<pseudo_core_kinetic_energy_density>)
        # - `SetupData.phi_jg` (<ae_partial_wave>)
        # - `SetupData.phit_jg` (<pseudo_partial_wave>)
        # - `SetupData.pt_jg` (<projector_function>)
        # - `SetupData.vt_g` (<pseudo_potential>)
        # - `SetupData.e_kin_jj` (<kinetic_energy_differences>)
        # - `SetupData.X_p` (<exact_exchange_X_matrix>)
        # - `SetupData.X_pg` (<yukawa_exchange_X_matrix>)
        setup.read_xml(fobj.read())
    return setup


def main(args: SimpleNamespace,
         gen: PAWSetupGenerator | None = None,
         plot: bool = True) -> None:
    setup = read_setup_file(args.paw)

    if args.create_basis_set in (True, False):
        basis_file = None
    else:
        basis_file = args.create_basis_set
        args.create_basis_set = True

    if gen is None and args.reconstruct_generator:
        gen = reconstruct_paw_gen(args.paw, basis_file)

    if gen and args.logarithmic_derivatives:
        fig_deriv = plt.figure()
        plot_log_derivs(gen,
                        args.logarithmic_derivatives,
                        plot=True,
                        ax=fig_deriv.gca())

    if not plot:
        return

    fig = plt.figure()
    basis: Basis | None

    if gen:
        subplots = fig.subplots(2, 2).flatten()
        gen.plot(potential_components=subplots[0],
                 partial_waves=subplots[1],
                 projectors=subplots[2])
        basis = gen.basis
    else:
        basis = read_basis_file(basis_file) if basis_file else None
        layout: tuple[int, ...]
        if basis:
            layout, i_ppw, i_pp = (2, 2), 1, 2
        else:
            layout, i_ppw, i_pp = (2,), 0, 1
        subplots = fig.subplots(*layout).flatten()
        plot_partial_waves(subplots[i_ppw], *get_ppw_params_setup_data(setup))
        plot_projectors(subplots[i_pp], *get_pp_params_setup_data(setup))

    if args.create_basis_set:
        if gen and basis is None:
            gen.create_basis_set()
            basis = gen.basis
            assert basis  # Assure `mypy` that it's a `Basis`
            basis.generatordata = ''  # we already printed this
        if basis:
            BasisPlotter().plot(basis, ax=subplots[-1])

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
            'if not provided, create a rudimentary basis set '
            '(requires `-r`)')
        add('-r', '--reconstruct-generator',
            action='store_true',
            help='Try to reconstruct the full PAW setup generator object; '
            'required for basis-set creation, and for plotting '
            'the potential components and logarithmic derivatives')
        add('-l', '--logarithmic-derivatives',
            metavar='spdfg,e1:e2:de,radius',
            help='Plot logarithmic derivatives (requires `-r`). ' +
            'Example: -l spdf,-1:1:0.05,1.3. ' +
            'Energy range and/or radius can be left out.')
        add('paw',
            metavar='DATASET',
            help='XML file from which to read the PAW dataset')

    @staticmethod
    def run(args):
        main(args)
