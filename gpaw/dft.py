from __future__ import annotations
from typing import Sequence, Type
from gpaw.new.brillouin import BZPoints
from functools import cached_property

_classes = {}


def register(cls):
    _classes[cls.__name__] = cls
    return cls


def ensure_object(obj, classes=_classes):
    if isinstance(obj, list):
        return [ensure_object((x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    if 'name' not in obj:
        return obj
    obj = obj.copy()
    cls = _classes[obj.pop('name')]
    return cls(**{key: ensure_object(value) for key, value in obj.items()})


class XC:
    def __init__(self, name):
        self.name = name

    def build(self, comm):
        return [comm, self.name]


class Mode:
    pass


@register
class PW(Mode):
    def __init__(self, *, ecut: float):
        self.ecut = ecut


class Eigensolver:
    pass


@register
class Davidson(Eigensolver):
    pass


class Extension:
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
                            for name, c in convergence.items()}
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

    def todict(self):
        dct = {}
        for key in self.non_defaults:
            value = self.__dict__[key]
            if hasattr(value, 'todict'):
                value = value.todict()
            dct[key] = value
        return dct
