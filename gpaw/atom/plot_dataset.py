from __future__ import annotations

import functools
import textwrap
from ast import literal_eval
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING
from xml.dom import minidom
from warnings import warn

from .. import typing
from ..basis_data import Basis, BasisPlotter
from ..setup_data import SetupData
from .aeatom import colors
from .generator2 import (PAWSetupGenerator, parameters,
                         generate, plot_log_derivs)
from .radialgd import AERadialGridDescriptor

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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


def plot_partial_waves(ax: 'Axes',
                       symbol: str,
                       name: str,
                       rgd: AERadialGridDescriptor,
                       cutoff: float,
                       iterator: Iterable[_PartialWaveItem]) -> None:
    r_g = rgd.r_g
    i = 0
    for l, n, rcut, e, phi_g, phit_g in sorted(iterator):
        if n == -1:
            gc = rgd.ceil(rcut)
            label = '*{} ({:.2f} Ha)'.format('spdf'[l], e)
        else:
            gc = len(rgd)
            label = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
        ax.plot(r_g[:gc], (phi_g * r_g)[:gc], color=colors[i], label=label)
        ax.plot(r_g[:gc], (phit_g * r_g)[:gc], '--', color=colors[i])
        i += 1
    ax.axis(xmin=0, xmax=3 * cutoff)
    ax.set_title(f'Partial waves: {symbol} {name}')
    ax.set_xlabel('radius [Bohr]')
    ax.set_ylabel(r'$r\phi_{n\ell}(r)$')
    ax.legend()


def plot_projectors(ax: 'Axes',
                    symbol: str,
                    name: str,
                    rgd: AERadialGridDescriptor,
                    cutoff: float,
                    iterator: Iterable[_ProjectorItem]) -> None:
    r_g = rgd.r_g
    i = 0
    for l, n, e, pt_g in sorted(iterator):
        if n == -1:
            label = '*{} ({:.2f} Ha)'.format('spdf'[l], e)
        else:
            label = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
        ax.plot(r_g, pt_g * r_g, color=colors[i], label=label)
        i += 1
    ax.axis(xmin=0, xmax=cutoff)
    ax.set_title(f'Projectors: {symbol} {name}')
    ax.set_xlabel('radius [Bohr]')
    ax.set_ylabel(r'$r\tilde{p}(r)$')
    ax.legend()


def _get_setup_symbol_and_name(setup: SetupData) -> tuple[str, str]:
    return setup.symbol, setup.setupname


def _get_gen_symbol_and_name(gen: PAWSetupGenerator) -> tuple[str, str]:
    aea = gen.aea
    return aea.symbol, aea.xc.name


def _get_setup_cutoff(setup: SetupData) -> float:
    cutoff = setup.r0
    if cutoff is not None:
        return cutoff

    # `.r0` can be `None` for 'old setups', whatever that means
    name = f'{setup.symbol}{setup.Nv}'
    params = parameters[name]
    if len(params) == 3:
        _, radii, extra = params
    else:
        _, radii = params
        extra = {}
    if 'r0' in extra:  # E.g. N5
        value = extra['r0']
        if TYPE_CHECKING:
            assert isinstance(value, float)
        return value
    if not isinstance(radii, Iterable):
        radii = [radii]
    return min(radii)


def get_ppw_params_paw_setup_generator(
    gen: PAWSetupGenerator,
) -> tuple[str, str,
           AERadialGridDescriptor, float,
           Iterable[_PartialWaveItem]]:
    return (*_get_gen_symbol_and_name(gen),
            gen.rgd,
            gen.rcmax,
            ((l, n, waves.rcut, e, phi_g, phit_g)
             for l, waves in enumerate(gen.waves_l)
             for n, e, phi_g, phit_g in zip(waves.n_n, waves.e_n,
                                            waves.phi_ng, waves.phit_ng)))


def get_ppw_params_setup_data(
    setup: SetupData,
) -> tuple[str, str,
           AERadialGridDescriptor, float,
           Iterable[_PartialWaveItem]]:
    return (*_get_setup_symbol_and_name(setup),
            setup.rgd,
            _get_setup_cutoff(setup),
            zip(setup.l_j, setup.n_j, setup.rcut_j, setup.eps_j,
                setup.phi_jg, setup.phit_jg))


def get_pp_params_paw_setup_generator(
    gen: PAWSetupGenerator,
) -> tuple[str, str, AERadialGridDescriptor, float, Iterable[_ProjectorItem]]:
    return (*_get_gen_symbol_and_name(gen),
            gen.rgd,
            gen.rcmax,
            ((l, n, e, pt_g)
             for l, waves in enumerate(gen.waves_l)
             for n, e, pt_g in zip(waves.n_n, waves.e_n, waves.pt_ng)))


def get_pp_params_setup_data(
    setup: SetupData,
) -> tuple[str, str, AERadialGridDescriptor, float, Iterable[_ProjectorItem]]:
    return (*_get_setup_symbol_and_name(setup),
            setup.rgd,
            _get_setup_cutoff(setup),
            zip(setup.l_j, setup.n_j, setup.eps_j, setup.pt_jg))


def reconstruct_paw_gen(setup: SetupData,
                        basis: Basis | None = None) -> PAWSetupGenerator:
    params = {'v0': None, **parse_generator_data(setup.generatordata)}
    gen = generate(**params)
    if basis is not None:
        gen.basis = basis
    return gen


def read_basis_file(basis: str) -> Basis:
    symbol, *chunks, end = basis.split('.')
    assert end == 'basis'
    name = '.'.join(chunks)
    return Basis.read_xml(symbol, name, basis)


def read_setup_file(dataset: str) -> SetupData:
    symbol, *_, setupname = dataset.split('.')
    setup = SetupData(symbol, setupname, readxml=False)
    with open(dataset, mode='rb') as fobj:
        setup.read_xml(fobj.read())
    if not setup.generatordata:
        generator, = minidom.parse(dataset).getElementsByTagName('generator')
        text, = generator.childNodes
        setup.generatordata = textwrap.dedent(text.data).strip('\n')
    return setup


def parse_generator_data(data: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for line in data.splitlines():
        key, sep, value = line.rstrip(',').partition('=')
        if not (sep and key.isidentifier()):
            continue
        try:
            value = literal_eval(value)
        except Exception:
            continue
        params[key] = value
    return params


def _get_figures_and_axes(
        ngraphs: int,
        separate_figures: bool = False) -> tuple[list['Figure'], list['Axes']]:
    from matplotlib import pyplot as plt

    if separate_figures:
        figs = []
        ax_objs = []
        for _ in range(ngraphs):
            fig = plt.figure()
            figs.append(fig)
            ax_objs.append(fig.gca())
        return figs, ax_objs

    assert ngraphs <= 6, f'Too many plots; expected <= 6, got {ngraphs}'
    if ngraphs > 4:
        layout = 2, 3
    elif ngraphs > 2:
        layout = 2, 2
    else:
        layout = 1, ngraphs

    fig = plt.figure()
    subplots = fig.subplots(*layout).flatten()  # type: ignore
    ntrimmed = layout[0] * layout[1] - ngraphs
    if ntrimmed:
        assert ntrimmed > 0, (f'Too many plots {ngraphs!r} '
                              f'for the layout {layout!r}')
        for ax in subplots[-ntrimmed:]:  # Remove unused subplots
            ax.remove()

    return [fig] * ngraphs, subplots[:ngraphs].tolist()


def plot_dataset(
    setup: SetupData,
    *,
    basis: Basis | None = None,
    gen: PAWSetupGenerator | None = None,
    plot_potential_components: bool = True,
    plot_partial_waves: bool = True,
    plot_projectors: bool = True,
    plot_logarithmic_derivatives: str | None = None,
    separate_figures: bool = False,
    reconstruct_generator: bool = False,
    savefig: str | None = None,
) -> tuple[list['Axes'], str | None]:
    """
    Return
    ------
    2-tuple: `tuple[list[Axes], <filename> | None]`
    """
    if gen is not None:
        reconstruct = False
    elif plot_logarithmic_derivatives or plot_potential_components:
        reconstruct = True
    else:
        reconstruct = False
    if reconstruct:
        data = setup.generatordata
        if parse_generator_data(data):
            gen = reconstruct_paw_gen(setup, basis)
        else:
            if data:
                data_status = 'malformed'
            else:
                data_status = 'missing'
            msg = ('cannot reconstruct the `PAWSetupGenerator` object '
                   f'({data_status} `setup.generatordata`), '
                   'so the logarithmic derivatives and/or '
                   'potential components cannot be plotted')
            warn(msg, stacklevel=2)
            plot_logarithmic_derivatives = None
            plot_potential_components = False

    plots: list[Callable] = []

    if gen is None:
        symbol, name, rgd, cutoff, ppw_iter = get_ppw_params_setup_data(setup)
        *_, pp_iter = get_pp_params_setup_data(setup)
    else:
        # TODO: maybe we can compare the `ppw_iter` and `pp_iter`
        # between the stored and regenerated values for verification
        (symbol, name,
         rgd, cutoff, ppw_iter) = get_ppw_params_paw_setup_generator(gen)
        *_, pp_iter = get_pp_params_paw_setup_generator(gen)

    if plot_logarithmic_derivatives:
        assert gen is not None
        plots.append(functools.partial(
            plot_log_derivs, gen, plot_logarithmic_derivatives, True))
    if plot_potential_components:
        assert gen is not None
        plots.append(gen.plot_potential_components)
    if plot_partial_waves:
        plots.append(functools.partial(
            # Name clash with local variable
            globals()['plot_partial_waves'],
            symbol=symbol, name=name, rgd=rgd, cutoff=cutoff,
            iterator=ppw_iter))
    if plot_projectors:
        plots.append(functools.partial(
            # Name clash with local variable
            globals()['plot_projectors'],
            symbol=symbol, name=name, rgd=rgd, cutoff=cutoff,
            iterator=pp_iter))

    if basis is not None:
        plots.append(functools.partial(BasisPlotter().plot, basis))

    if savefig is not None:
        separate_figures = False
    figs, ax_objs = _get_figures_and_axes(len(plots), separate_figures)
    assert len(figs) == len(ax_objs) == len(plots)
    for ax, plot_func in zip(ax_objs, plots):
        plot_func(ax=ax)

    if savefig is not None:
        assert len({id(fig) for fig in figs}) == 1
        fig, *_ = figs
        assert fig is not None
        fig.savefig(savefig)

    return ax_objs, savefig


def main(args: SimpleNamespace) -> None:
    from matplotlib import pyplot as plt

    setup = read_setup_file(args.dataset)
    basis = None if args.basis_set is None else read_basis_file(args.basis_set)
    sep_figs = args.outfile is None and args.separate_figures
    ax_objs, fname = plot_dataset(
        setup,
        basis=basis,
        separate_figures=sep_figs,
        plot_potential_components=args.potential_components,
        plot_logarithmic_derivatives=args.logarithmic_derivatives,
        savefig=args.outfile)
    assert ax_objs

    if fname is None:
        plt.show()


class CLICommand:
    """Plot the PAW dataset,
    which by default includes the partial waves and the projectors.
    """
    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('-b', '--basis-set',
            metavar='FILE',
            help='Load and plot the basis set from an XML file')
        add('-p', '--potential-components',
            action='store_true',
            help='Plot the potential components '
            '(this reconstructs the full PAW setup generator object)')
        add('-l', '--logarithmic-derivatives',
            metavar='spdfg,e1:e2:de,radius',
            help='Plot logarithmic derivatives '
            '(this reconstructs the full PAW setup generator object). '
            'Example: -l spdf,-1:1:0.05,1.3. '
            'Energy range and/or radius can be left out.')
        add('-s', '--separate-figures',
            action='store_true',
            help='If not plotting to a file, '
            'plot the plots in separate figure windows/tabs, '
            'instead of as subplots/panels in the same figure')
        add('-o', '--outfile', '--write',
            metavar='FILE',
            help='Write the plots to FILE instead of `plt.show()`-ing them')
        add('dataset',
            metavar='FILE',
            help='XML file from which to read the PAW dataset')

    @staticmethod
    def run(args):
        main(args)
