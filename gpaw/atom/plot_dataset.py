from __future__ import annotations

import functools
import textwrap
import os
from ast import literal_eval
from collections.abc import Callable, Iterable
from math import pi
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING
from xml.dom import minidom
from warnings import warn

from .. import typing
from ..basis_data import Basis, BasisPlotter
from ..setup_data import SetupData, read_maybe_unzipping, search_for_file
from .aeatom import AllElectronAtom, colors
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
    for l, n, rcut, e, phi_g, phit_g in sorted(iterator):
        if n == -1:
            gc = rgd.ceil(rcut)
            label = '*{} ({:.2f} Ha)'.format('spdf'[l], e)
        else:
            gc = len(rgd)
            label = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
        color = colors[l]
        ax.plot(r_g[:gc], (phi_g * r_g)[:gc], color=color, label=label)
        ax.plot(r_g[:gc], (phit_g * r_g)[:gc], '--', color=color)
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
    for l, n, e, pt_g in sorted(iterator):
        if n == -1:
            label = '*{} ({:.2f} Ha)'.format('spdf'[l], e)
        else:
            label = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
        ax.plot(r_g, pt_g * r_g, color=colors[l], label=label)
    ax.axis(xmin=0, xmax=cutoff)
    ax.set_title(f'Projectors: {symbol} {name}')
    ax.set_xlabel('radius [Bohr]')
    ax.set_ylabel(r'$r\tilde{p}(r)$')
    ax.legend()


def plot_potential_components(ax: 'Axes',
                              symbol: str,
                              name: str,
                              rgd: AERadialGridDescriptor,
                              cutoff: float,
                              components: dict[str, typing.Array1D]) -> None:
    assert components
    radial_grid = rgd.r_g
    for color, (key, label) in zip(
            colors,
            [('xc', 'xc'), ('zero', '0'), ('hamiltonian', 'H'),
             ('pseudo', 'ps'), ('all_electron', 'ae')]):
        if key in components:
            ax.plot(radial_grid, components[key], color=color, label=label)
    arrays = components.values()
    ax.axis(xmin=0,
            xmax=2 * cutoff,
            ymin=min(array[1:].min() for array in arrays),
            ymax=max(0, *(array[1:].max() for array in arrays)))
    ax.set_yscale('symlog')
    ax.set_title(f'Potential components: {symbol} {name}')
    ax.set_xlabel('radius [Bohr]')
    ax.set_ylabel('potential [Ha]')
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


def get_pc_params_paw_setup_generator(
    gen: PAWSetupGenerator,
) -> tuple[str, str, AERadialGridDescriptor, float, dict[str, typing.Array1D]]:
    assert gen.vtr_g is not None  # Appease `mypy`

    rgd = gen.rgd
    r_g = rgd.r_g
    nan = float('nan')
    zero = rgd.zeros()
    zero[0] = nan
    zero[1:] = gen.v0r_g[1:] / r_g[1:]
    hamiltonian = rgd.zeros()
    hamiltonian[0] = nan
    hamiltonian[1:] = gen.vHtr_g[1:] / r_g[1:]
    pseudo = rgd.zeros()
    pseudo[0] = nan
    pseudo[1:] = gen.vtr_g[1:] / r_g[1:]
    all_electron = rgd.zeros()
    all_electron[0] = nan
    all_electron[1:] = gen.aea.vr_sg[0, 1:] / r_g[1:]
    components = {'xc': gen.vxct_g,
                  'zero': zero,
                  'hamiltonian': hamiltonian,
                  'pseudo': pseudo,
                  'all_electron': all_electron}
    return (*_get_gen_symbol_and_name(gen), rgd, gen.rcmax, components)


def get_pc_params_setup_data(
    setup: SetupData,
) -> tuple[str, str, AERadialGridDescriptor, float, dict[str, typing.Array1D]]:
    prefactor = (4 * pi) ** -.5
    zero = setup.vbar_g * prefactor
    if setup.vt_g is None:
        pseudo = None
    else:
        pseudo = setup.vt_g * prefactor
    symbol, xc_name = _get_setup_symbol_and_name(setup)
    rgd = setup.rgd
    # Reconstruct the AEA object
    # (Note: this misses the empty bound states from projectors)
    aea = AllElectronAtom(symbol, xc_name,
                          Z=setup.Z,
                          configuration=list(zip(setup.n_j, setup.l_j,
                                                 setup.f_j, setup.eps_j)))
    if setup.has_corehole:
        aea.add(setup.ncorehole, setup.lcorehole, -setup.fcorehole)
    aea.initialize(rgd.N)
    aea.run()
    aea.scalar_relativistic = setup.type == 'scalar-relativistic'
    aea.refine()
    all_electron = rgd.zeros()
    all_electron[0] = float('nan')
    all_electron[1:] = aea.vr_sg[0, 1:] / rgd.r_g[1:]
    # Note: the XC and Hamiltonian parts cannot be extracted from the
    # setup data
    components = {'zero': zero, 'all_electron': all_electron}
    if pseudo is not None:
        components['pseudo'] = pseudo
    return (symbol, xc_name, rgd, _get_setup_cutoff(setup), components)


def reconstruct_paw_gen(setup: SetupData,
                        basis: Basis | None = None) -> PAWSetupGenerator:
    params = {'v0': None, **parse_generator_data(setup.generatordata)}
    gen = generate(**params)
    if basis is not None:
        gen.basis = basis
    return gen


def read_basis_file(basis: str) -> Basis:
    symbol, *chunks, end = os.path.basename(basis).split('.')
    if end == 'gz':
        *chunks, end = chunks
    assert end == 'basis'
    name = '.'.join(chunks)
    if not os.path.isfile(basis):
        basis, _ = search_for_file(basis)
    return Basis.read_xml(symbol, name, basis)


def read_setup_file(dataset: str) -> SetupData:
    symbol, *name, xc = os.path.basename(dataset).split('.')
    if xc == 'gz':
        *name, xc = name
    if os.path.isfile(dataset):
        setup = SetupData(symbol, xc, readxml=False)
        setup.read_xml(read_maybe_unzipping(dataset))
    else:
        setup = SetupData.find_and_read_path(symbol, xc,
                                             '.'.join(name) or 'paw')
        dataset = setup.filename
    if not setup.generatordata:
        generator, = (minidom.parseString(read_maybe_unzipping(dataset))
                      .getElementsByTagName('generator'))
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
                   '(some of) the potential components cannot be plotted')
            warn(msg, stacklevel=2)
            plot_logarithmic_derivatives = None

    plots: list[Callable] = []

    if gen is None:
        symbol, name, rgd, cutoff, ppw_iter = get_ppw_params_setup_data(setup)
        *_, pp_iter = get_pp_params_setup_data(setup)
        *_, pot_comps = get_pc_params_setup_data(setup)
    else:
        # TODO: maybe we can compare the `ppw_iter` and `pp_iter`
        # between the stored and regenerated values for verification
        (symbol, name,
         rgd, cutoff, ppw_iter) = get_ppw_params_paw_setup_generator(gen)
        *_, pp_iter = get_pp_params_paw_setup_generator(gen)
        *_, pot_comps = get_pc_params_paw_setup_generator(gen)

    if plot_logarithmic_derivatives:
        assert gen is not None
        plots.append(functools.partial(
            plot_log_derivs, gen, plot_logarithmic_derivatives, True))
    if plot_potential_components:
        plots.append(functools.partial(
            # Name clash with local variable
            globals()['plot_potential_components'],
            symbol=symbol, name=name, rgd=rgd, cutoff=cutoff,
            components=pot_comps))
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
