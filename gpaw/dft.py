from __future__ import annotations
from typing import Sequence, Type
from gpaw.new.brillouin import BZPoints
from functools import cached_property
from dataclasses import dataclass, is_dataclass, asdict

_classes = {}


def register(cls):
    _classes[cls.__name__.lower()] = cls
    return dataclass(cls)


def ensure_object(obj, classes=_classes):
    if isinstance(obj, list):
        return [ensure_object(x, classes) for x in obj]
    if not isinstance(obj, dict):
        return obj
    if 'name' not in obj:
        return obj
    obj = obj.copy()
    cls = _classes[obj.pop('name')]
    return cls(**{key: ensure_object(value, classes)
                  for key, value in obj.items()})


class XC:
    def __init__(self, name):
        self.name = name

    def build(self, comm):
        return [comm, self.name]


class Mode:
    pass


@register
class PW(Mode):
    ecut: float = 340


class Eigensolver:
    pass


@register
class Davidson(Eigensolver):
    pass


class Extension:
    pass


class Mixer:
    pass


class Occupations:
    pass


class PoissonSolver:
    pass


class Symmetry:
    pass


class Parameters:
    def __init__(
        self,
        *,
        mode: str | dict | Mode,
        charge: float = 0.0,
        convergence: dict | None = None,
        eigensolver: dict | Eigensolver = Eigensolver(),
        gpts: Sequence[int] | None = None,
        h: float = -1.0,
        hund: bool = False,
        basis: str | dict[str | int | None, str] = '',
        extensions: list[Extension] | None = None,
        kpts: Sequence[int] | dict | BZPoints = (1, 1, 1),
        magmoms: Sequence[float] | Sequence[Sequence[float]] | None = None,
        maxiter: int = 333,
        mixer: dict | Mixer = Mixer(),
        nbands: int | str = '',
        occupations: dict | Occupations = Occupations(),
        poissonsolver: dict | PoissonSolver = PoissonSolver(),
        random: bool = False,
        setups: str | dict = '',
        soc: bool = False,
        spinpol: bool = False,
        symmetry: str | dict | Symmetry = 'on',
        xc: str | dict | XC = 'LDA',
        hooks: dict[str, Type] | None = None):
        """DFT-parameters object.

        """

        classes = _classes | (hooks or {})

        def obj(o):
            return ensure_object(o, classes)

        self.mode = obj(mode)
        self.charge = charge
        self.convergence = {name: obj(c)
                            for name, c in (convergence or {}).items()}
        self.eigensolver = obj(eigensolver)

    def __repr__(self):
        lines = []
        for key in self.non_defaults:
            value = self.__dict__[key]
            lines.append(f'{key}={value!r}')
        return ',\n'.join(lines)

    @cached_property
    def non_defaults(self):
        nd = ['mode']
        if self.charge:
            nd.append('charge')
        return nd

    def todict(self, everthing=False):
        dct = {}
        for key in self.non_defaults:
            value = self.__dict__[key]
            if is_dataclass(value):
                name = value.__class__.__name__.lower()
                value = {'name': name} | asdict(value)
            dct[key] = value
        return dct


if __name__ == '__main__':
    p = Parameters(xc='PBE', mode=PW(ecut=200))
    print(p)
    print(p.todict())
    p = Parameters(**p.todict())
    print(p)
