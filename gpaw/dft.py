from __future__ import annotations

import warnings
from pathlib import Path
from typing import IO, TYPE_CHECKING, Iterable, Sequence, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets
from gpaw.mpi import MPIComm
from gpaw.new.gpw import read_gpw
from gpaw.new.logger import Logger
from gpaw.new.symmetry import Symmetries, create_symmetries_object
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    from gpaw.new.ase_interface import ASECalculator

PARAMETER_NAMES = [
    'mode', 'basis', 'charge', 'convergence', 'eigensolver',
    'experimental', 'gpts', 'h', 'hund', 'extensions', 'kpts',
    'magmoms', 'maxiter', 'mixer', 'nbands', 'occupations',
    'parallel', 'poissonsolver', 'random', 'setups', 'soc',
    'spinpol', 'symmetry', 'xc']


class DeprecatedParameterWarning(FutureWarning):
    """Warning class for when a parameter or its value is deprecated."""


class XC:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def functional(self, collinear):
        from gpaw.xc import XC as xc
        return xc({'name': self.name, **self.kwargs},
                  collinear=collinear)

    @classmethod
    def from_param(cls, xc):
        if isinstance(xc, str):
            xc = {'name': xc}
        return XC(**xc)


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
    def __init__(self,
                 *,
                 force_complex_dtype: bool = False):
        self.force_complex_dtype = force_complex_dtype

    @classmethod
    def from_param(cls, mode) -> Mode:
        if isinstance(mode, str):
            mode = {'name': mode}
        if isinstance(mode, dict):
            mode = mode.copy()
            return {'pw': PW}[mode.pop('name')](**mode)
        return mode


class PW(Mode):
    def __init__(self,
                 ecut: float = 340,
                 *,
                 qspiral=None,
                 dedecut=None,
                 dtype: DTypeLike | None = None,
                 force_complex_dtype: bool = False):
        self.ecut = ecut
        self.dtype = dtype
        self.qspiral = qspiral
        self.dedecut = dedecut
        super().__init__(force_complex_dtype=force_complex_dtype)

    def todict(self):
        return {'ecut': self.ecut,
                **self._not_none('dtype', 'qspiral', 'dedecut')}

    def dft_components_builder(self, atoms, params, *, log=None, comm=None):
        from gpaw.new.pw.builder import PWDFTComponentsBuilder
        return PWDFTComponentsBuilder(atoms, params, log=log, comm=comm)


class LCAO(Mode):
    pass


class Eigensolver(Parameter):
    @classmethod
    def from_param(cls, eigensolver):
        if isinstance(eigensolver, str):
            return eigensolvers[eigensolver]()
        if 'name' in eigensolver:
            eigensolver = eigensolver.copy()
            return eigensolvers[eigensolver.pop('name')](**eigensolver)
        return DefaultEigensolver(eigensolver)


class DefaultEigensolver(Eigensolver):
    def __init__(self, params: dict):
        self.params = params

    def todict(self):
        return self.params


class Davidson(Eigensolver):
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
    def build_lcao(self, basis, relpos_ac, cell_cv, symmetries):
        return LCAOEigensolver(basis)


class HybridLCAOEigensolver(LCAOEigensolver):
    def build_lcao(self, basis, relpos_ac, cell_cv, symmetries):
        from gpaw.new.lcao.hybrids import HybridLCAOEigensolver as HLCAOES
        return HLCAOES(basis, relpos_ac, cell_cv)


class Scissors(LCAOEigensolver):
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


class Extension:
    pass


class Mixer(Parameter):
    def __init__(self, params: dict):
        self.params = params

    def todict(self):
        return self.params

    @classmethod
    def from_param(cls, mixer):
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
        return Symmetry(**(s or {}))

    def todict(self):
        return ...

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


class KPoints(Parameter):
    @classmethod
    def from_param(cls, kpts):
        if isinstance(kpts, KPoints):
            return kpts
        if isinstance(kpts, dict):
            kpts = kpts.copy()
            kpts.pop('name', '')
        return MonkhorstPack.from_param(kpts)


class MonkhorstPack(KPoints):
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
            dct['gama'] = self.gamma
        return dct

    @classmethod
    def from_param(cls,
                   kpts: Sequence[int] | dict | MonkhorstPack
                   ) -> MonkhorstPack:
        if isinstance(kpts, MonkhorstPack):
            return kpts
        if isinstance(kpts, dict):
            return MonkhorstPack(**kpts)
        return MonkhorstPack(size=kpts)

    def build(self, atoms):
        from gpaw.new.brillouin import MonkhorstPackKPoints
        size, offset = kpts2sizeandoffsets(**self.todict(), atoms=atoms)
        for n, periodic in zip(size, atoms.pbc):
            if not periodic and n != 1:
                raise ValueError('K-points can only be used with PBCs!')
        return MonkhorstPackKPoints(size, offset)


DOCS = """
mode:
    PW, LCAO or FD mode.
basis:
    Basis-set.
"""


class Parameters:
    def __init__(
        self,
        *,
        mode: str | dict | Mode,
        basis: str | dict[str | int | None, str] = '',
        charge: float = 0.0,
        convergence: dict | None = None,
        eigensolver: str | dict | Eigensolver | None = None,
        gpts: Sequence[int] | None = None,
        h: float = 0.0,
        hund: bool = False,
        experimental: dict | None = None,
        extensions: Sequence[Extension] = (),
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
        xc: str | dict | XC = 'LDA'):
        """DFT-parameters object.

        >>> p = Parameters(mode=PW(400))
        >>> p
        >>> p.charge
        0.0
        >>> p.xc
        >>> atoms = Atoms()
        >>> dft = p.dft_calculation(atoms)
        >>> atoms.calc = p.ase_calculator(atoms)

        """ + DOCS

        self._non_defaults = []
        for key, value in locals().items():
            if key in ['gpts', 'kpts', 'magmoms']:
                is_default = value is None
            elif key == 'xc':
                is_default = value == 'LDA'
            elif key != 'self':
                is_default = not value
            else:
                continue
            if not is_default:
                self._non_defaults.append(key)

        if h != 0.0 and gpts is not None:
            raise ValueError("""You can't use both "gpts" and "h"!""")

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
        self.mode = Mode.from_param(mode)
        self.basis = {None: basis} if isinstance(basis, str) else basis
        self.charge = charge
        self.convergence = convergence or {}
        self.eigensolver = Eigensolver.from_param(eigensolver or {})
        self.gpts = np.array(gpts) if gpts is not None else None
        self.h = h
        self.hund = hund
        self.experimental = experimental or {}
        self.extensions = list(extensions or [])
        self.kpts = KPoints.from_param(kpts or (1, 1, 1))
        self.magmoms = np.array(magmoms) if magmoms is not None else None
        self.maxiter = maxiter
        self.mixer = Mixer.from_param(mixer or {})
        self.nbands = nbands if nbands != '' else 'default'
        self.occupations = Occupations.from_param(occupations or {})
        self.parallel = parallel or {}
        self.poissonsolver = PoissonSolver.from_param(poissonsolver or {})
        self.random = random
        self.setups = {None: setups} if isinstance(setups, str) else setups
        self.soc = soc
        self.spinpol = spinpol
        self.symmetry = Symmetry.from_param(symmetry)
        self.xc = XC.from_param(xc)
        _fix_legacy_stuff(self)

    def __repr__(self):
        lines = []
        for key in self._non_defaults:
            value = self.__dict__[key]
            lines.append(f'{key}={value!r}')
        return ',\n'.join(lines)

        # txt = pformat(val, width=75 - n).replace('\n', '\n ' + ' ' * n)

    def todict(self, everything=False):
        dct = {}
        if everything:
            keys = (key for key in self.__dict__ if key[0] != '_')
        else:
            keys = self._non_defaults
        for key in keys:
            value = self.__dict__[key]
            if hasattr(value, 'todict'):
                name = value.__class__.__name__.lower()
                value = {'name': name} | value.todict()
            elif key == 'basis':
                if list(value) == [None]:
                    value = value[None]
            dct[key] = value
        return dct

    def dft_component_builder(self, atoms, *, comm=None, log=None):
        return self.mode.dft_components_builder(
            atoms, self, comm=comm, log=log)

    def dft_calculation(self,
                        atoms,
                        txt: str = '-',
                        communicator=None):
        from gpaw.new.calculation import DFTCalculation
        log = Logger(txt, communicator)
        return DFTCalculation.from_parameters(atoms, self, log.comm, log)

    def dft_info(self, atoms):
        ...


def DFT(
    atoms,
    *,
    mode,
    basis: str | dict[str | int | None, str] = '',
    charge: float = 0.0,
    convergence: dict | None = None,
    eigensolver: dict | Eigensolver | None = None,
    experimental: dict | None = None,
    gpts: Sequence[int] | None = None,
    h: float = 0.0,
    hund: bool = False,
    extensions: Sequence[Extension] = (),
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
    txt: str | Path | IO[str] | None = '?',
    communicator: MPIComm | Iterable[int] | None = None):
    """asdg
    """
    params = Parameters(**{k: v for k, v in locals().items()
                           if k in PARAMETER_NAMES})
    return params.dft_calculation(atoms, txt, communicator)


def GPAW(
    filename: Union[str, Path, IO[str]] = None,
    *,
    basis: str | dict[str | int | None, str] = '',
    charge: float = 0.0,
    convergence: dict | None = None,
    eigensolver: dict | Eigensolver | None = None,
    gpts: Sequence[int] | None = None,
    h: float = 0.0,
    hund: bool = False,
    experimental: dict | None = None,
    extensions: Sequence[Extension] = (),
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
    communicator: MPIComm | Iterable[int] | None = None,
    hooks: dict | None = None) -> ASECalculator:
    """Create ASE-compatible GPAW calculator.

    """
    from gpaw.new.ase_interface import ASECalculator

    if txt == '?':
        txt = '-' if filename is None else None

    log = Logger(txt, communicator)

    if mode == '':
        del mode

    kwargs = {key: value for key, value in locals().items()
              if key in PARAMETER_NAMES}

    if filename is not None:
        args = Parameters(mode='pw', **kwargs).non_defaults
        if set(args) >= {'mode', 'parallel'}:
            raise ValueError(
                'Illegal argument(s) when reading from a file: '
                f'{", ".join(args)}')
        atoms, dft, params, _ = read_gpw(filename,
                                         log=log,
                                         parallel=parallel,
                                         hooks=hooks)
        return ASECalculator(params,
                             log=log, dft=dft, atoms=atoms)

    params = Parameters(**kwargs)
    return ASECalculator(params, log=log)


def _fix_legacy_stuff(params):
    if not isinstance(params.mode, Mode):
        params.mode = Mode.from_param(params.mode.todict())
