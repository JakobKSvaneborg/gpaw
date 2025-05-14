from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Sequence, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets
from numpy.typing import DTypeLike
from gpaw.mpi import MPIComm
from gpaw.new.logger import Logger
from gpaw.new.symmetry import Symmetries, create_symmetries_object

if TYPE_CHECKING:
    from gpaw.new.ase_interface import ASECalculator

PARAMETER_NAMES = [
    'mode', 'basis', 'charge', 'convergence', 'eigensolver', 'environment',
    'experimental', 'extensions', 'gpts', 'h', 'hund',
    'interpolation', 'kpts', 'magmoms', 'maxiter', 'mixer', 'nbands',
    'occupations', 'parallel', 'poissonsolver', 'random', 'setups', 'soc',
    'spinpol', 'symmetry', 'xc']


class DeprecatedParameterWarning(FutureWarning):
    """Warning class for when a parameter or its value is deprecated."""


class Parameter:
    def __repr__(self):
        args = ', '.join(f'{k}={v!r}' for k, v in self.todict().items())
        return f'{self.__class__.__name__}({args})'

    def _not_none(self, *keys: str) -> dict:
        dct = {}
        for key in keys:
            value = self.__dict__[key]
            if value is not None:
                dct[key] = value
        return dct


class Mode(Parameter):
    qspiral = None

    def __init__(self,
                 *,
                 dtype: DTypeLike | None = None,
                 force_complex_dtype: bool = False):
        self.dtype = dtype
        self.force_complex_dtype = force_complex_dtype
        self.name = self.__class__.__name__.lower()

    def todict(self) -> dict:
        dct = self._not_none('dtype')
        if self.force_complex_dtype:
            dct['force_complex_dtype'] = True
        return dct

    @classmethod
    def from_param(cls, mode) -> Mode:
        if isinstance(mode, str):
            mode = {'name': mode}
        if isinstance(mode, dict):
            mode = mode.copy()
            return {'pw': PW,
                    'lcao': LCAO,
                    'fd': FD,
                    'tb': TB}[mode.pop('name')](**mode)
        return mode

    def dft_components_builder(self, atoms, params, *, log=None, comm=None):
        module = importlib.import_module(f'gpaw.new.{self.name}.builder')
        return getattr(module, f'{self.name.upper()}DFTComponentsBuilder')(
            atoms, params, log=log, comm=comm)


class PW(Mode):
    def __init__(self,
                 ecut: float = 340,
                 *,
                 qspiral=None,
                 dedecut=None,
                 dtype: DTypeLike | None = None,
                 force_complex_dtype: bool = False):
        self.ecut = ecut
        self.qspiral = qspiral
        self.dedecut = dedecut
        super().__init__(dtype=dtype,
                         force_complex_dtype=force_complex_dtype)

    def todict(self):
        dct = super().todict()
        dct |= self._not_none('ecut', 'qspiral', 'dedecut')
        return dct


class LCAO(Mode):
    distribution = '?'

    def __init__(self,
                 *,
                 dtype: DTypeLike | None = None,
                 force_complex_dtype: bool = False):
        super().__init__(dtype=dtype,
                         force_complex_dtype=force_complex_dtype)


class FD(Mode):
    def __init__(self,
                 *,
                 nn=3,
                 dtype: DTypeLike | None = None,
                 force_complex_dtype: bool = False):
        self.nn = nn
        super().__init__(dtype=dtype,
                         force_complex_dtype=force_complex_dtype)

    def todict(self):
        dct = super().todict()
        if self.nn != 3:
            dct['nn'] = self.nn
        return dct


class TB(Mode):
    distribution = '?'


class Eigensolver(Parameter):
    @classmethod
    def from_param(cls, eigensolver):
        if isinstance(eigensolver, str):
            eigensolver = {'name': eigensolver}
        elif not isinstance(eigensolver, dict):
            return eigensolver
        if 'name' in eigensolver:
            eigensolver = eigensolver.copy()
            name = eigensolver.pop('name')
            if name == 'dav':
                name = 'davidson'
                warnings.warn('Please use "davidson" instead of "dav"')
            if name in eigensolvers:
                return eigensolvers[name](**eigensolver)
            raise ValueError(f'Unknown eigensolver: {name}')
        return DefaultEigensolver(eigensolver)


class DefaultEigensolver(Eigensolver):
    def __init__(self, params: dict):
        self.params = params

    def todict(self):
        return self.params


class Davidson(Eigensolver):
    name = 'davidson'

    def __init__(self, niter: int = 2):
        self.niter = niter

    def todict(self):
        return {'niter': self.niter}

    def build(self,
              nbands,
              wf_desc,
              band_comm,
              create_preconditioner,
              converge_bands,
              setups,
              atoms):
        from gpaw.new.pwfd.davidson import Davidson
        return Davidson(
            nbands,
            wf_desc,
            band_comm,
            create_preconditioner,
            converge_bands,
            niter=self.niter)


class RMMDIIS(Eigensolver):
    name = 'rmm-diis'

    def __init__(self, niter: int = 2):
        self.niter = niter

    def todict(self):
        return {'niter': self.niter}

    def build(self,
              nbands,
              wf_desc,
              band_comm,
              create_preconditioner,
              converge_bands,
              setups,
              atoms):
        from gpaw.new.pwfd.rmmdiis import RMMDIIS
        return RMMDIIS(
            nbands,
            wf_desc,
            band_comm,
            create_preconditioner,
            converge_bands,
            niter=self.niter)


class LCAOEigensolver(Eigensolver):
    name = 'lcao'

    def build_lcao(self, basis, relpos_ac, cell_cv, symmetries):
        from gpaw.new.lcao.eigensolver import LCAOEigensolver as LCAOES
        return LCAOES(basis)


class HybridLCAOEigensolver(LCAOEigensolver):
    def build_lcao(self, basis, relpos_ac, cell_cv, symmetries):
        from gpaw.new.lcao.hybrids import HybridLCAOEigensolver as HLCAOES
        return HLCAOES(basis, relpos_ac, cell_cv)


class Scissors(LCAOEigensolver):
    name = 'scissors'

    def __init__(self, shifts: list):
        self.shifts = shifts

    def todict(self):
        return {'shifts': self.shifts}

    def build_lcao(self, basis, relpos_ac, cell_cv, symmetries):
        from gpaw.lcao.scissors import ScissorsLCAOEigensolver
        return ScissorsLCAOEigensolver(basis,
                                       self.shifts,
                                       symmetries)


eigensolvers = {
    'davidson': Davidson,
    'rmm-diis': RMMDIIS,
    'lcao': LCAOEigensolver,
    'hybrid-lcao': HybridLCAOEigensolver,
    'scissors': Scissors}


class Extension(Parameter):
    @classmethod
    def from_param(self, extension):
        if isinstance(extension, dict):
            dct = extension.copy()
            name = dct.pop('name')
            if name == 'd3':
                from gpaw.new.extensions import D3
                return D3(**dct)
            1 / 0
        return extension


class Environment(Parameter):
    @classmethod
    def from_param(self, env):
        if env is None:
            return Environment()
        if isinstance(env, dict):
            dct = env.copy()
            name = dct.pop('name')
            if name == 'sjm':
                from gpaw.new.sjm import SJM
                return SJM(**dct)
            if name == 'solvation':
                from gpaw.new.solvation import Solvation
                return Solvation(**dct)
            raise ValueError(f'Unknown environment: {name}')
        return env

    def build(self,
              setups,
              grid,
              relpos_ac,
              log,
              comm):
        from gpaw.new.environment import Environment as Env
        return Env(len(setups))


class Mixer(Parameter):
    def __init__(self, params: dict):
        self.params = params

    def todict(self):
        return self.params

    @classmethod
    def from_param(cls, mixer):
        if isinstance(mixer, Mixer):
            return mixer
        return Mixer(mixer)


class Occupations(Parameter):
    def __init__(self, params: dict):
        self.params = params

    def todict(self):
        return self.params

    @classmethod
    def from_param(cls, occupations):
        if isinstance(occupations, dict):
            return Occupations(occupations)
        return occupations


class PoissonSolver(Parameter):
    def __init__(self, params: dict):
        self.params = params

    def todict(self):
        return self.params

    @classmethod
    def from_param(cls, ps):
        if isinstance(ps, dict):
            return PoissonSolver(ps)
        return ps

    def build(self, *, grid, xp=np):
        from gpaw.poisson import PoissonSolver as make_poisson_solver
        solver = make_poisson_solver(**self.params, xp=xp)
        return solver.build(grid, xp)


def array_or_none(a):
    if a is None:
        return None
    return np.array(a)


class Symmetry(Parameter):
    def __init__(self,
                 *,
                 rotations: np.ndarray | None = None,
                 translations: np.ndarray | None = None,
                 atommaps: np.ndarray | None = None,
                 extra_ids: Sequence[int] | None = None,
                 tolerance: float | None = None,  # Å
                 point_group: bool = True,
                 symmorphic: bool = True,
                 time_reversal: bool = True):
        self.rotations = array_or_none(rotations)
        self.translations = array_or_none(translations)
        self.atommaps = array_or_none(atommaps)
        self.extra_ids = array_or_none(extra_ids)
        self.tolerance = tolerance
        self.point_group = point_group
        self.symmorphic = symmorphic
        self.time_reversal = time_reversal

    @classmethod
    def from_param(cls, s):
        if isinstance(s, Symmetry):
            return s
        if isinstance(s, str):
            if s == 'off':
                return Symmetry(point_group=False, time_reversal=False)
            if s == 'on':
                return Symmetry()
            raise ValueError()
        if 'name' in s:
            s = s.copy()
            del s['name']
        return Symmetry(**(s or {}))

    def todict(self):
        dct = self._not_none('rotations', 'translations', 'atommaps',
                             'extra_ids', 'tolerance')
        for name in ['point_group', 'symmorphic', 'time_reversal']:
            if not getattr(self, name):
                dct[name] = False
        return dct

    def build(self,
              atoms: Atoms,
              *,
              setup_ids: Sequence | None = None,
              magmoms: np.ndarray | None = None,
              _backwards_compatible=False) -> Symmetries:
        return create_symmetries_object(
            atoms,
            setup_ids=setup_ids,
            magmoms=magmoms,
            rotations=self.rotations,
            translations=self.translations,
            atommaps=self.atommaps,
            extra_ids=self.extra_ids,
            tolerance=self.tolerance,
            point_group=self.point_group,
            symmorphic=self.symmorphic,
            _backwards_compatible=_backwards_compatible)


class BZSampling(Parameter):
    @classmethod
    def from_param(cls, kpts):
        if isinstance(kpts, BZSampling):
            return kpts
        if isinstance(kpts, dict):
            if 'kpts' in kpts:
                return KPoints(kpts['kpts'])
            kpts = kpts.copy()
            kpts.pop('name', '')
        else:
            kpts = np.array(kpts)
            if kpts.ndim == 1:
                kpts = {'size': kpts}
            else:
                return KPoints(kpts)
        return MonkhorstPack(**kpts)


class KPoints(BZSampling):
    def __init__(self,
                 kpts: Sequence[Sequence[float]]):
        self.kpts = kpts

    def todict(self):
        return {'kpts': self.kpts}

    def build(self, atoms):
        from gpaw.new.brillouin import BZPoints
        return BZPoints(self.kpts)


class MonkhorstPack(BZSampling):
    def __init__(self,
                 size: Sequence[int] | None = None,
                 density: float | None = None,
                 gamma: bool | None = None):
        self.size = size
        self.density = density
        self.gamma = gamma

    def todict(self):
        dct = {}
        if self.size is not None:
            dct['size'] = self.size
        if self.density is not None:
            dct['density'] = self.density
        if self.gamma is not None:
            dct['gamma'] = self.gamma
        return dct

    def build(self, atoms):
        from gpaw.new.brillouin import MonkhorstPackKPoints
        size, offset = kpts2sizeandoffsets(**self.todict(), atoms=atoms)
        for n, periodic in zip(size, atoms.pbc):
            if not periodic and n != 1:
                raise ValueError('K-points can only be used with PBCs!')
        return MonkhorstPackKPoints(size, offset)


class XC(Parameter):
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def todict(self):
        return {'name': self.name, **self.kwargs}

    def functional(self, collinear):
        from gpaw.xc import XC as xc
        return xc({'name': self.name, **self.kwargs},
                  collinear=collinear)

    @classmethod
    def from_param(cls, xc):
        if isinstance(xc, XC):
            return xc
        if isinstance(xc, str):
            xc = {'name': xc}
        return XC(**xc)


KptsType = Union[Sequence[int], dict, Sequence[Sequence[float]]]


class Parameters:
    def __init__(
        self,
        *,
        mode: str | dict | Mode,
        basis: str | dict[str | int | None, str] = '',
        charge: float = 0.0,
        convergence: dict | None = None,
        eigensolver: str | dict | Eigensolver | None = None,
        environment=None,
        gpts: Sequence[int] | None = None,
        h: float = 0.0,
        hund: bool = False,
        experimental: dict | None = None,
        extensions: Sequence[Extension] = (),
        interpolation: int = 0,
        kpts: KptsType | MonkhorstPack | None = None,
        magmoms: Sequence[float] | Sequence[Sequence[float]] | None = None,
        maxiter: int = 0,
        mixer: dict | Mixer | None = None,
        nbands: int | str = '',
        occupations: dict | Occupations | None = None,
        parallel: dict | None = None,
        poissonsolver: dict | PoissonSolver | None = None,
        random: bool = False,
        setups: str | dict = '',
        soc: bool = False,
        spinpol: bool = False,
        symmetry: str | dict | Symmetry = '',
        xc: str | dict | XC = 'LDA'):
        """DFT-parameters object.

        >>> p = Parameters(mode=PW(400))
        >>> p
        mode=PW(ecut=400)
        >>> p.charge
        0.0
        >>> p.xc
        XC(name='LDA')
        >>> from ase.build import molecule
        >>> atoms = molecule('H2', vacuum=3.0)
        >>> dft = p.dft_calculation(atoms, txt='h2.txt')
        >>> atoms.calc = dft.ase_calculator()

        Parameters
        ==========
        mode:
            PW, LCAO or FD mode.
        basis:
            Basis-set.
        """

        if experimental is None:
            experimental = {}
        else:
            experimental = experimental.copy()
        if experimental.pop('niter_fixdensity', None) is not None:
            warnings.warn('Ignoring "niter_fixdensity".')
        if 'reuse_wfs_method' in experimental:
            del experimental['reuse_wfs_method']
            warnings.warn('Ignoring "reuse_wfs_method".')
        if 'soc' in experimental:
            warnings.warn('Please use new "soc" parameter.',
                          DeprecatedParameterWarning)
            soc = experimental.pop('soc')
        if 'magmoms' in experimental:
            warnings.warn('Please use new "magmoms" parameter.',
                          DeprecatedParameterWarning)
            magmoms = experimental.pop('magmoms')
        unknown = experimental.keys() - {'backwards_compatible'}
        if unknown:
            warnings.warn(f'Unknown experimental keyword(s): {unknown}',
                          stacklevel=3)

        self._non_defaults = []
        _locals = locals()
        for key in PARAMETER_NAMES:
            value = _locals[key]
            if key in ['gpts', 'kpts', 'magmoms']:
                is_default = value is None
            elif key == 'xc':
                is_default = value == 'LDA'
            else:
                is_default = not value
            if not is_default:
                self._non_defaults.append(key)

        if h != 0.0 and gpts is not None:
            raise ValueError("""You can't use both "gpts" and "h"!""")

        self.mode = Mode.from_param(mode)
        basis = basis or {}
        self.basis = ({'default': basis} if not isinstance(basis, dict)
                      else basis)
        self.charge = charge
        self.convergence = convergence or {}
        self.eigensolver = Eigensolver.from_param(eigensolver or {})
        self.environment = Environment.from_param(environment)
        self.gpts = np.array(gpts) if gpts is not None else None
        self.h = h
        self.hund = hund
        self.experimental = experimental or {}
        self.extensions = [Extension.from_param(ext) for ext in extensions]
        self.interpolation = interpolation
        self.kpts = BZSampling.from_param((1, 1, 1) if kpts is None else kpts)
        self.magmoms = np.array(magmoms) if magmoms is not None else None
        self.maxiter = maxiter
        self.mixer = Mixer.from_param(mixer or {})
        self.nbands = nbands if nbands != '' else 'default'
        self.occupations = Occupations.from_param(occupations or {})
        self.parallel = parallel or {}
        self.poissonsolver = PoissonSolver.from_param(poissonsolver or {})
        self.random = random
        setups = setups or 'paw'
        self.setups = ({'default': setups} if isinstance(setups, str)
                       else setups)
        self.soc = soc
        self.spinpol = spinpol
        self.symmetry = Symmetry.from_param(symmetry or 'on')
        self.xc = XC.from_param(xc)
        _fix_legacy_stuff(self)

        for key in self.parallel:
            if key not in PARALLEL_KEYS:
                raise ValueError(
                    f'Unknown key: {key!r}.  '
                    f'Must be one of {", ".join(PARALLEL_KEYS)}')

    def __repr__(self):
        lines = []
        for key in self._non_defaults:
            value = self._value(key)
            lines.append(f'{key}={value!r}')
        return ',\n'.join(lines)

    @property
    def kwargs(self):
        return {key: self.__dict__[key] for key in self._non_defaults}

    def todict(self):
        dct = {}
        for key in self._non_defaults:
            value = self._value(key)
            if hasattr(value, 'todict'):
                name = getattr(value, 'name', None)
                value = value.todict()
                if name is not None:
                    value['name'] = name
            dct[key] = value
        return dct

    def _value(self, key: str) -> Any:
        value = self.__dict__[key]
        if key == 'basis':
            if list(value) == [None]:
                value = value[None]
        return value

    def dft_component_builder(self, atoms, *, comm=None, log=None):
        return self.mode.dft_components_builder(
            atoms, self, comm=comm, log=log)

    def dft_calculation(self,
                        atoms,
                        txt: str | Path | IO[str] | None = '-',
                        communicator: MPIComm | Sequence[int] | None = None):
        from gpaw.new.calculation import DFTCalculation
        log = Logger(txt, communicator)
        return DFTCalculation.from_parameters(atoms, self, log.comm, log)

    def dft_info(self, atoms):
        ...


PARALLEL_KEYS = {
    'kpt', 'domain', 'band', 'order', 'stridebands', 'augment_grids',
    'sl_auto', 'sl_default', 'sl_diagonalize', 'sl_inverse_cholesky',
    'sl_lcao', 'sl_lrtddft', 'use_elpa', 'elpasolver', 'buffer_size', 'gpu'}


def DFT(
    atoms,
    *,
    mode,
    basis: str | dict[str | int | None, str] = '',
    charge: float = 0.0,
    convergence: dict | None = None,
    eigensolver: dict | Eigensolver | None = None,
    environment=None,
    experimental: dict | None = None,
    gpts: Sequence[int] | None = None,
    h: float = 0.0,
    hund: bool = False,
    extensions: Sequence[Extension] = (),
    interpolation: int = 0,
    kpts: Sequence[int] | dict | MonkhorstPack | None = None,
    magmoms: Sequence[float] | Sequence[Sequence[float]] | None = None,
    maxiter: int = 0,
    mixer: dict | Mixer | None = None,
    nbands: int | str = '',
    occupations: dict | Occupations | None = None,
    parallel: dict | None = None,
    poissonsolver: dict | PoissonSolver | None = None,
    random: bool = False,
    setups: str | dict = '',
    soc: bool = False,
    spinpol: bool = False,
    symmetry: str | dict | Symmetry = '',
    xc: str | dict | XC = 'LDA',
    txt: str | Path | IO[str] | None = '-',
    communicator: MPIComm | Sequence[int] | None = None):
    """asdg
    """
    params = Parameters(**{k: v for k, v in locals().items()
                           if k in PARAMETER_NAMES})
    return params.dft_calculation(atoms, txt, communicator)


def GPAW(
    filename: str | Path | IO[str] | None = None,
    *,
    basis: str | dict[str | int | None, str] = '',
    charge: float = 0.0,
    convergence: dict | None = None,
    eigensolver: dict | Eigensolver | None = None,
    environment=None,
    gpts: Sequence[int] | None = None,
    h: float = 0.0,
    hund: bool = False,
    experimental: dict | None = None,
    extensions: Sequence[Extension] = (),
    interpolation: int = 0,
    kpts: Sequence[int] | dict | MonkhorstPack | None = None,
    magmoms: Sequence[float] | Sequence[Sequence[float]] | None = None,
    maxiter: int = 0,
    mixer: dict | Mixer | None = None,
    mode: str | dict | Mode = '',
    nbands: int | str = '',
    occupations: dict | Occupations | None = None,
    parallel: dict | None = None,
    poissonsolver: dict | PoissonSolver | None = None,
    random: bool = False,
    setups: str | dict = '',
    soc: bool = False,
    spinpol: bool = False,
    symmetry: str | dict | Symmetry = '',
    xc: str | dict | XC = 'LDA',
    txt: str | Path | IO[str] | None = '?',
    communicator: MPIComm | Sequence[int] | None = None,
    object_hooks=None) -> ASECalculator:
    """Create ASE-compatible GPAW calculator.

    """
    from gpaw.new.ase_interface import ASECalculator
    from gpaw.new.gpw import read_gpw

    if txt == '?':
        txt = '-' if filename is None else None

    log = Logger(txt, communicator)

    if mode == '':
        del mode

    kwargs = {key: value for key, value in locals().items()
              if key in PARAMETER_NAMES}

    if filename is not None:
        args = Parameters(mode='pw', **kwargs)._non_defaults
        if set(args) > {'mode', 'parallel'}:
            raise ValueError(
                'Illegal argument(s) when reading from a file: '
                f'{", ".join(args)}')
        atoms, dft, params, _ = read_gpw(filename,
                                         log=log,
                                         parallel=parallel,
                                         object_hooks=object_hooks)
        return ASECalculator(params,
                             log=log, dft=dft, atoms=atoms)

    params = Parameters(**kwargs)
    return ASECalculator(params, log=log)


def _fix_legacy_stuff(params):
    if not isinstance(params.mode, Mode):
        dct = params.mode.todict()
        if 'interpolation' in dct:
            params.interpolation = dct.pop('interpolation')
        params.mode = Mode.from_param(dct)
    if not isinstance(params.eigensolver, Eigensolver):
        params.eigensolver = Eigensolver.from_param(
            params.eigensolver.todict())
    if not isinstance(params.mixer, Mixer):
        params.mixer = Mixer.from_param(params.mixer.todict())
