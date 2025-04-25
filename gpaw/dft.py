from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import IO, Any, Callable, Iterable, Protocol, Sequence, Union

import numpy as np

from gpaw.mpi import MPIComm, broadcast, synchronize_atoms, world

PARAMETER_NAMES = [
    'mode',
    'basis',
    'charge',
    'convergence',
    'eigensolver',
    'gpts',
    'h',
    'hund',
    'extensions',
    'kpts',
    'magmoms',
    'maxiter',
    'mixer',
    'nbands',
    'occupations',
    'poissonsolver',
    'random',
    'setups',
    'soc',
    'spinpol',
    'symmetry',
    'xc']


@dataclass
class XC:
    name: str

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


@dataclass
class PW(Mode):
    ecut: float = 340


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


class Symmetry:
    @classmethod
    def from_param(cls, s):
        if s is None:
            return Symmetry()


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


class Missing:
    pass


missing = Missing()

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
    poissonsolver: dict | PoissonSolver | None = None,
    random: bool = False,
    setups: str | dict = '',
    soc: bool = False,
    spinpol: bool = False,
    symmetry: str | dict | Symmetry = '',
    xc: str | dict | XC = 'LDA',
    txt: str | Path | IO[str] | None = '?',
    communicator: MPIComm | Iterable[int] | None = None,
    parallel: dict[str, Any] | None = None):
    """asdg
    """
    params = Parameters(**{k: v for k, v in locals().items()
                           if k in PARAMETER_NAMES})
    return params.dft_calculation(atoms, txt, communicator, parallel)


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
        extensions: Sequence[Extension] = (),
        kpts: Sequence[int] | dict | MonkhorstPack | None = None,
        magmoms: Sequence[float] | Sequence[Sequence[float]] | None = None,
        maxiter: int = 0,
        mixer: dict | Mixer | None = None,
        nbands: int | str = '',
        occupations: dict | Occupations | None = None,
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
        self.extensions = list(extensions or [])
        self.kpts = KPoints.from_param(kpts)
        self.magmoms = np.array(magmoms) if magmoms is not None else None
        self.maxiter = maxiter
        self.mixer = Mixer.from_param(mixer),
        self.nbands = nbands
        self.occupations = Occupations.from_param(occupations)
        self.poissonsolver = PoissonSolver.from_param(poissonsolver)
        self.random = random
        self.setups = {None: setups} if isinstance(setups, str) else setups
        self.soc = soc
        self.spinpol = spinpol
        self.symmetry = Symmetry.from_param(symmetry)
        self.xc = XC.from_param(xc)

    def __repr__(self):
        lines = []
        for key in self._non_defaults:
            value = self.__dict__[key]
            lines.append(f'{key}={value!r}')
        return ',\n'.join(lines)

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
                        txt = '-',
                        communicator=None,
                        parallel=None):
        from gpaw.new.calculation import DFTCalculation
        log = Logger(txt, communicator)
        parallel???
        return DFTCalculation.from_parameters(atoms, self, log.comm, log)


if __name__ == '__main__':
    p = Parameters(xc='PBE', mode=PW(ecut=200), charge=1.0)
    print(p)
    print(p.todict())
    p = Parameters(**p.todict())
    print(p)
