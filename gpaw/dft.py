from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Iterable, Sequence, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets

from gpaw.mpi import MPIComm
from gpaw.new.gpw import read_gpw
from gpaw.new.logger import Logger
from gpaw.new.symmetry import Symmetries, create_symmetries_object

if TYPE_CHECKING:
    from gpaw.new.ase_interface import ASECalculator

PARAMETER_NAMES = [
    'mode', 'basis', 'charge', 'convergence', 'eigensolver',
    'experimental', 'gpts', 'h', 'hund', 'extensions', 'kpts',
    'magmoms', 'maxiter', 'mixer', 'nbands', 'occupations',
    'parallel', 'poissonsolver', 'random', 'setups', 'soc',
    'spinpol', 'symmetry', 'xc']


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


class Mode:
    @classmethod
    def from_param(cls, mode) -> Mode:
        if isinstance(mode, str):
            mode = {'name': mode}
        if isinstance(mode, dict):
            mode = mode.copy()
            return {'pw': PW}[mode.pop('name')](**mode)
        return mode

    @property
    def dft_components_builder_class(self):
        name = self.__class__.__name__
        mod = importlib.import_module(f'gpaw.new.{name.lower()}.builder')
        return getattr(mod, f'{name}DFTComponentsBuilder')


@dataclass
class PW(Mode):
    ecut: float = 340


class LCAO(Mode):
    pass


@dataclass
class Eigensolver:
    parameters: dict

    @classmethod
    def from_param(cls, eigensolver):
        if eigensolver is None:
            eigensolver = {}
        if 'name' in eigensolver:
            eigensolver = eigensolver.copy()
            return {'dav': Davidson}[eigensolver.pop('name')](**eigensolver)
        return Eigensolver(eigensolver)


@dataclass
class Davidson(Eigensolver):
    niter: int = 2


class Extension:
    pass


class Mixer:
    @classmethod
    def from_param(cls, mixer):
        if mixer is None:
            return Mixer()


class Occupations:
    @classmethod
    def from_param(cls, occupations):
        if occupations is None:
            return Occupations()


class PoissonSolver:
    @classmethod
    def from_param(cls, ps):
        if ps is None:
            return PoissonSolver()


@dataclass
class Symmetry:
    rotations: np.ndarray | None = None
    translations: np.ndarray | None = None
    atommaps: np.ndarray | None = None
    extra_ids: Sequence[int] | None = None
    tolerance: float | None = None  # Å
    point_group: bool = True
    symmorphic: bool = True
    time_reversal: bool = True

    @classmethod
    def from_param(cls, s):
        return Symmetry(**(s or {}))

    def build(self,
              atoms: Atoms,
              *,
              setup_ids: Sequence | None = None,
              magmoms: np.ndarray | None = None,
              _backwards_compatible=False) -> Symmetries:
        kwargs = asdict(self)
        del kwargs['time_reversal']
        return create_symmetries_object(
            atoms, **kwargs, _backwards_compatible=_backwards_compatible)


class KPoints:
    @classmethod
    def from_param(cls, kpts):
        if isinstance(kpts, KPoints):
            return kpts
        if isinstance(kpts, dict):
            kpts = kpts.copy()
            kpts.pop('name', '')
        return MonkhorstPack.from_param(kpts)


@dataclass
class MonkhorstPack(KPoints):
    size: Sequence[int] | None = None
    density: float | None = None
    gamma: bool | None = None

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
        size, offset = kpts2sizeandoffsets(**asdict(self), atoms=atoms)
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


class Parameters:
    def __init__(
        self,
        *,
        mode: str | dict | Mode,
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

        self.mode = Mode.from_param(mode)
        self.basis = {None: basis} if isinstance(basis, str) else basis
        self.charge = charge
        self.convergence = convergence or {}
        self.eigensolver = Eigensolver.from_param(eigensolver)
        self.gpts = np.array(gpts) if gpts is not None else None
        self.h = h
        self.hund = hund
        self.experimental = experimental or {}
        self.extensions = list(extensions or [])
        self.kpts = KPoints.from_param(kpts or (1, 1, 1))
        self.magmoms = np.array(magmoms) if magmoms is not None else None
        self.maxiter = maxiter
        self.mixer = Mixer.from_param(mixer),
        self.nbands = nbands if nbands != '' else 'default'
        self.occupations = Occupations.from_param(occupations)
        self.parallel = parallel or {}
        self.poissonsolver = PoissonSolver.from_param(poissonsolver)
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
            if is_dataclass(value):
                name = value.__class__.__name__.lower()
                value = {'name': name} | asdict(value)
            dct[key] = value
        return dct

    def dft_calculation(self,
                        atoms,
                        txt: str = '-',
                        communicator=None):
        from gpaw.new.calculation import DFTCalculation
        log = Logger(txt, communicator)
        return DFTCalculation.from_parameters(atoms, self, log.comm, log)


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

    kwargs = {key: value for key, value in locals().items()
              if key in PARAMETER_NAMES}

    if filename is not None:
        if mode is not None:
            raise ValueError('"mode" not allowed when reading from a file')
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


if __name__ == '__main__':
    p = Parameters(xc='PBE', mode=PW(ecut=200), charge=1.0)
    print(p)
    print(p.todict())
    p = Parameters(**p.todict())
    print(p)
