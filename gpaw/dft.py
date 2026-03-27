from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Union, Literal, Dict

import numpy as np
from numpy.typing import DTypeLike

from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets

from gpaw import GPAW_NEW
from gpaw.mpi import MPIComm
from gpaw.new.calculation import DFTCalculation
from gpaw.new.logger import Logger
from gpaw.new.pwfd.davidson import Davidson as DavidsonEigensolver
from gpaw.new.pwfd.ppcg import PPCG as PPCGEigensolver
from gpaw.new.pwfd.rmmdiis import RMMDIIS as RMMDIISEigensolver
from gpaw.new.symmetry import Symmetries, create_symmetries_object

if TYPE_CHECKING:
    from gpaw.new.ase_interface import ASECalculator

PARAMETER_NAMES = [
    'mode', 'basis', 'charge', 'convergence', 'eigensolver',
    'experimental', 'extensions', 'gpts', 'h', 'hund',
    'interpolation', 'kpts', 'magmoms', 'maxiter', 'mixer', 'nbands',
    'occupations', 'parallel', 'poissonsolver', 'random', 'setups', 'soc',
    'spinpol', 'symmetry', 'xc',
    # for old GPAW:
    'background_charge', 'external']


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
                 dtype: DTypeLike = 'double',
                 force_complex_dtype: bool = False,
                 interpolation=117):
        # Translate legacy dtypes
        if dtype == 'float32' or dtype == np.float32:
            dtype = 'single'
            force_complex_dtype = False
        elif dtype == 'float64' or dtype == np.float64:
            dtype = 'double'
            force_complex_dtype = False
        elif dtype == 'complex64' or dtype == np.complex64:
            dtype = 'single'
            force_complex_dtype = True
        elif dtype == 'complex128' or dtype == np.complex128:
            dtype = 'double'
            force_complex_dtype = True

        self.dtype = dtype
        self.force_complex_dtype = force_complex_dtype
        self.name = self.__class__.__name__.lower()
        if interpolation != 117:
            raise LegacyGPAWError

    def todict(self) -> dict:
        dct: Dict[str, Any] = {}
        if self.force_complex_dtype:
            dct['force_complex_dtype'] = True
        if self.dtype == 'single':
            dct['dtype'] = 'single'
        return dct

    @classmethod
    def from_param(cls, mode) -> Mode:
        if isinstance(mode, Mode):
            return mode
        if isinstance(mode, str):
            mode = {'name': mode}
        elif not isinstance(mode, dict):
            mode = mode.todict()
        mode = mode.copy()
        return {'pw': PW,
                'lcao': LCAO,
                'fd': FD,
                'tb': TB}[mode.pop('name')](**mode)

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
                 dtype: DTypeLike = 'double',
                 force_complex_dtype: bool = False,
                 **kwargs):
        """PW-mode.

        Parameters
        ==========
        ecut:
            Plane-wave cutoff energy in eV.
        """
        if 'interpolation' in kwargs:
            raise LegacyGPAWError

        self.ecut = ecut
        self.qspiral = qspiral
        self.dedecut = dedecut
        super().__init__(dtype=dtype,
                         force_complex_dtype=force_complex_dtype,
                         **kwargs)

    def todict(self):
        dct = super().todict()
        dct |= self._not_none('ecut', 'qspiral', 'dedecut')
        return dct


class LCAO(Mode):
    distribution = '?'


class FD(Mode):
    def __init__(self,
                 *,
                 nn=3,
                 dtype: str = 'double',
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
        from gpaw.new.do import DirectOptimization
        from gpaw.new.eigensolver import Eigensolver as NewEigensolver
        from gpaw.old.eigensolvers.eigensolver import Eigensolver as OES

        eigensolvers = {
            'davidson': Davidson,
            'rmm-diis': RMMDIIS,
            'etdm-fdpw': DirectOptimization,
            'ppcg': PPCG,
            'lcao': LCAOEigensolver,
            'hybrid-lcao': HybridLCAOEigensolver,
            'scissors': Scissors}

        match eigensolver:
            case str(name):
                return cls.from_param({'name': name})
            case {'name': name, **kwargs}:
                if name == 'dav':
                    warnings.warn('Please use "davidson" instead of "dav"')
                    return eigensolvers['davidson'](**kwargs)
                if name in eigensolvers:
                    return eigensolvers[name](**kwargs)
                if GPAW_NEW == 147:
                    raise LegacyGPAWError
                raise ValueError(f'Unknown name of eigensolver: {name}')
            case {**kwargs}:
                return DefaultEigensolver(kwargs)
            case NewEigensolver():
                return eigensolver
            case OES():
                return cls.from_param(eigensolver.todict())
            case _:
                if GPAW_NEW == 147:
                    raise LegacyGPAWError
                raise ValueError(f'Unknown eigensolver input: {eigensolver}')


class DefaultEigensolver(Eigensolver):
    def __init__(self, params: dict):
        self.params = params

    def todict(self):
        return self.params


class PWFDEigensolverParameter(Eigensolver):
    def __init__(self,
                 niter: int = 2,
                 max_buffer_mem: int = 200 * 1024**2):
        self.niter = niter
        self.max_buffer_mem = max_buffer_mem

    def todict(self):
        return {'niter': self.niter}

    def build(self,
              nbands,
              wf_desc,
              band_comm,
              domain_band_comm,
              scalapack_parameters,
              hamiltonian,
              convergence,
              setups,
              atoms):
        return self.cls(
            nbands,
            wf_desc,
            band_comm,
            domain_band_comm,
            hamiltonian,
            convergence,
            niter=self.niter,
            scalapack_parameters=scalapack_parameters,
            max_buffer_mem=self.max_buffer_mem)


class Davidson(PWFDEigensolverParameter):
    name = 'davidson'
    cls = DavidsonEigensolver


class PPCG(PWFDEigensolverParameter):
    name = 'ppcg'
    cls = PPCGEigensolver

    def __init__(self,
                 niter: int = 2,
                 min_niter: int | None = 1,
                 max_buffer_mem: int = 200 * 1024**2,
                 blocksize=None,
                 rr_modulo=5,
                 include_cg=True,
                 promote_inner_dtype=False,
                 tolerances: tuple[float, float, float] | None = None):
        self.niter = niter
        self.min_niter = min_niter
        self.max_buffer_mem = max_buffer_mem
        self.blocksize = blocksize
        self.rr_modulo = rr_modulo
        self.include_cg = include_cg
        self.promote_inner_dtype = promote_inner_dtype
        self.tolerances = tolerances

    def todict(self):
        return {'niter': self.niter,
                'min_niter': self.min_niter,
                'max_buffer_mem': self.max_buffer_mem,
                'blocksize': self.blocksize,
                'rr_modulo': self.rr_modulo,
                'include_cg': self.include_cg,
                'promote_inner_dtype': self.promote_inner_dtype,
                'tolerances': self.tolerances}

    def build(self,
              nbands,
              wf_desc,
              band_comm,
              domain_band_comm,
              scalapack_parameters,
              hamiltonian,
              convergence,
              setups,
              atoms):
        return self.cls(
            nbands,
            wf_desc,
            band_comm,
            hamiltonian,
            convergence,
            domain_band_comm=domain_band_comm,
            niter=self.niter,
            min_niter=self.min_niter,
            max_buffer_mem=self.max_buffer_mem,
            blocksize=self.blocksize,
            rr_modulo=self.rr_modulo,
            include_cg=self.include_cg,
            promote_inner_dtype=self.promote_inner_dtype,
            tolerances=self.tolerances)


class RMMDIIS(PWFDEigensolverParameter):
    name = 'rmm-diis'
    cls = RMMDIISEigensolver

    def __init__(self,
                 niter: int = 1,
                 diis_steps: int = 1,
                 max_buffer_mem: int = 200 * 1024**2,
                 trial_step: float | None = None):
        self.niter = niter
        self.diis_steps = diis_steps
        self.max_buffer_mem = max_buffer_mem
        self.trial_step = trial_step

    def todict(self):
        return {'niter': self.niter,
                'diis_steps': self.diis_steps,
                'max_buffer_mem': self.max_buffer_mem,
                'trial_step': self.trial_step}

    def build(self,
              nbands,
              wf_desc,
              band_comm,
              domain_band_comm,
              scalapack_parameters,
              create_preconditioner,
              convergence,
              setups,
              atoms):
        return self.cls(
            nbands,
            wf_desc,
            band_comm,
            create_preconditioner,
            convergence,
            domain_band_comm=domain_band_comm,
            scalapack_parameters=scalapack_parameters,
            niter=self.niter,
            diis_steps=self.diis_steps,
            max_buffer_mem=self.max_buffer_mem,
            trial_step=self.trial_step)


class LCAOEigensolver(Eigensolver):
    name = 'lcao'

    def build_lcao(self, basis, relpos_ac, cell_cv, symmetries):
        from gpaw.new.lcao.eigensolver import LCAOEigensolver as LCAOES
        return LCAOES(basis)

    def todict(self):
        return {}


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


class ExtensionInput(Parameter):
    @classmethod
    def from_input(self, extension: ExtensionInput | dict):
        if isinstance(extension, dict):
            dct = extension.copy()
            name = dct.pop('name')
            if name == 'd3':
                from gpaw.new.extensions import D3
                return D3(**dct)
            if name == 'spin_direction_constraint':
                from gpaw.new.constraints import SpinDirectionConstraint
                return SpinDirectionConstraint(**dct)
            if name == 'sjm':
                from gpaw.new.sjm import SJM
                return SJM(**dct)
            if name == 'solvation':
                from gpaw.new.solvation import Solvation
                return Solvation(**dct)
            raise ValueError(f'Unknown extension: {name}')
        return extension


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
            rotation_scc=self.rotations,
            translation_sc=self.translations,
            atommap_sa=self.atommaps,
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
        if hasattr(kpts, 'kpts'):
            return KPoints(kpts.kpts)
        if isinstance(kpts, dict):
            if 'kpts' in kpts:
                return KPoints(kpts['kpts'])
            if 'path' in kpts:
                return BandPath(**kpts)
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
                 gamma: bool | None = None,
                 even: bool | None = None):
        self.size = size
        self.density = density
        self.gamma = gamma
        self.even = even

    def todict(self):
        dct = {}
        if self.size is not None:
            dct['size'] = self.size
        if self.density is not None:
            dct['density'] = self.density
        if self.gamma is not None:
            dct['gamma'] = self.gamma
        if self.even is not None:
            dct['even'] = self.even
        return dct

    def build(self, atoms):
        from gpaw.new.brillouin import MonkhorstPackKPoints
        size, offset = kpts2sizeandoffsets(**self.todict(), atoms=atoms)
        for n, periodic in zip(size, atoms.pbc):
            if not periodic and n != 1:
                raise ValueError('K-points can only be used with PBCs!')
        return MonkhorstPackKPoints(size, offset)


class BandPath(BZSampling):
    def __init__(self,
                 path: str,
                 npoints: int):
        self.path = path
        self.npoints = npoints

    def todict(self):
        return {'path': self.path, 'npoints': self.npoints}

    def build(self, atoms):
        from gpaw.new.brillouin import BZBandPath
        return BZBandPath(atoms.cell.bandpath(self.path,
                                              npoints=self.npoints,
                                              pbc=atoms.pbc))


class XC(Parameter):
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def todict(self):
        return {'name': self.name, **self.kwargs}

    def functional(self, *, collinear: bool, atoms: Atoms | None = None):
        from gpaw.xc import XC as xc
        return xc({'name': self.name, **self.kwargs},
                  collinear=collinear, atoms=atoms)

    @classmethod
    def from_param(cls, xc):
        if isinstance(xc, XC):
            return xc
        if isinstance(xc, str):
            xc = {'name': xc}
        if not isinstance(xc, dict):
            raise LegacyGPAWError
        return XC(**xc)


KptsType = Union[Sequence[int], dict, Sequence[Sequence[float]]]

PARALLEL_KEYS = {
    'kpt', 'domain', 'band', 'order', 'stridebands', 'augment_grids',
    'sl_auto', 'sl_default', 'sl_diagonalize', 'sl_inverse_cholesky',
    'sl_lcao', 'sl_lrtddft', 'use_elpa', 'elpasolver', 'buffer_size', 'gpu'}


class Parameters:
    def __init__(
        self,
        *,
        mode: str | dict | Mode,
        basis: str | dict[str | int | None, str] | None = None,
        charge: float | None = None,
        convergence: dict | None = None,
        eigensolver: str | dict | Eigensolver | None = None,
        experimental: dict | None = None,
        extensions: Sequence[ExtensionInput] | None = None,
        gpts: Sequence[int] | None = None,
        h: float | None = None,
        hund: bool | None = None,
        interpolation: int | Literal['fft'] | None = None,
        kpts: KptsType | MonkhorstPack | None = None,
        magmoms: Sequence[float] | Sequence[Sequence[float]] | None = None,
        maxiter: int | None = None,
        mixer: dict | Mixer | None = None,
        nbands: int | str | None = None,
        occupations: dict | Occupations | None = None,
        parallel: dict | None = None,
        poissonsolver: dict | PoissonSolver | None = None,
        random: bool | None = None,
        setups: str | dict | None = None,
        soc: bool | None = None,
        spinpol: bool | None = None,
        symmetry: str | dict | Symmetry | None = None,
        xc: str | dict | XC | None = None,
        external=None):
        r"""DFT-parameters object.

        >>> p = Parameters(mode=PW(400))
        >>> p
        mode=PW(ecut=400)
        >>> p.charge
        0.0
        >>> p.xc
        XC(name='LDA')
        >>> from ase.build import molecule
        >>> atoms = molecule('H2', vacuum=3.0)
        >>> dft = p.dft_calculation(atoms, txt=None)
        >>> atoms.calc = dft.ase_calculator()

        Parameters
        ==========
        mode:
            PW, LCAO or FD mode.
        basis:
            Basis-set.  Used for LCAO calculations and wave-function initial
            guess for PW and FD calculations.  Default is to use the PAW
            pseudo partial-waves.
        charge:
            Total charge of the system in units of `|e|`.
        convergence:
            SCF-convergence criteria.
        eigensolver:
            Eigensolver.  Default for PW and FD mode is ``'davidson'``.
        gpts:
            Number of real-space grid-points for wave-functions
            (three integers).
        h:
            Grid-spacing for wave-function grid (Å).  Default value is
            0.2 Å for LCAO and FD mode calculations.  For a PW-mode
            calculation, we use the formula `h=γh_0` with `γ \simeq 1.4` and:

            .. math::

               h_0 = \frac{\pi}{\sqrt{8E_c}}.

            Ideally, we would use `\gamma=1`, but in practice, 1.4 is
            a good compromise between accuracy and efficiency.
            In eV and Å units we have `h_0=3.07/\sqrt{E_c}`.
        hund:
            Use Hund's rule for initial magnetic moments.
        experimental:
            Experimental stuff.
        extensions:
            Extensions (D3, ...).
        interpolation:
            ...
        kpts:
            Brillouin-zone sampling.  Default is Γ-point only.
        magmoms:
            Initial magnetic moments for non-collinear calculations.
        maxiter:
            Maximum number of allowed SCF-iterations.  Default is 333.
        mixer:
            Density-mixing scheme.
        nbands:
            Number of bands.
        occupations:
            ...
        parallel:
            Parallelization strategy.  Example:  Force parallelization
            over ``'kpt`` with ``{'band': 1, 'domain': 1}``.
        poissonsolver:
            ...
        random:
            Use random numbers for initial wave functions.
        setups:
            ...
        soc:
            Enable spin-orbit coupling.
        spinpol:
            Force spin-polarized calculation.
        symmetry:
            Use of symmetry.  Default is to use ...
        xc:
            XC-functional.  Default is PZ-LDA.
        """
        if external is not None:
            raise LegacyGPAWError
        soc, magmoms = _parse_experimental(experimental, soc, magmoms)
        self._non_defaults = [
            key for key, value in locals().items()
            if value is not None and key != 'self']

        if h is not None and gpts is not None:
            raise ValueError("""You can't use both "gpts" and "h"!""")

        self.mode = Mode.from_param(mode)
        basis = basis or {}
        self.basis = ({'default': basis} if not isinstance(basis, dict)
                      else basis)
        self.charge = charge or 0.0
        self.convergence = convergence or {}
        self.eigensolver = Eigensolver.from_param(eigensolver or {})
        self.experimental = experimental or {}
        self.extensions = [ExtensionInput.from_input(ext)
                           for ext in extensions or []]
        self.gpts = np.array(gpts) if gpts is not None else None
        self.h = h
        self.hund = hund or False
        self.interpolation = interpolation
        self.kpts = BZSampling.from_param((1, 1, 1) if kpts is None else kpts)
        self.magmoms = np.array(magmoms) if magmoms is not None else None
        self.maxiter = maxiter or 333
        self.mixer = Mixer.from_param(mixer or {})
        self.nbands = nbands
        self.occupations = Occupations.from_param(occupations or {})
        self.parallel = parallel or {}
        self.poissonsolver = PoissonSolver.from_param(poissonsolver or {})
        self.random = random or False
        setups = setups or 'paw'
        self.setups = ({'default': setups} if isinstance(setups, str)
                       else setups)
        self.soc = soc or False
        self.spinpol = spinpol or False
        self.symmetry = Symmetry.from_param(symmetry or 'on')
        self.xc = XC.from_param(xc or 'LDA')

        _fix_legacy_stuff(self)

        for key in self.parallel:
            if key not in PARALLEL_KEYS:
                raise ValueError(
                    f'Unknown key: {key!r}.  '
                    f'Must be one of {", ".join(PARALLEL_KEYS)}')

    def __repr__(self) -> str:
        lines = []
        for key in self._non_defaults:
            value = self._value(key)
            lines.append(f'{key}={value!r}')
        return ',\n'.join(lines)

    def todict(self) -> dict:
        dct = {}
        for key in self._non_defaults:
            value = self._value(key)
            if hasattr(value, 'todict'):
                name = getattr(value, 'name', None)
                value = value.todict()
                if name is not None:
                    value['name'] = name
            elif key == 'extensions':
                value = [{'name': x.name, **x.todict()}
                         for x in self.extensions]
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
                        communicator: MPIComm | None = None
                        ) -> DFTCalculation:
        log = Logger(txt, communicator)
        return DFTCalculation.from_parameters(atoms, self, log.comm, log)

    def dft_info(self, atoms):
        ...


def _parse_experimental(experimental: dict | None,
                        soc: bool | None,
                        magmoms) -> tuple:
    if experimental is None:
        return soc, magmoms
    if experimental.pop('niter_fixdensity', None) is not None:
        warnings.warn('Ignoring "niter_fixdensity".')
    if 'reuse_wfs_method' in experimental:
        del experimental['reuse_wfs_method']
        warnings.warn('Ignoring "reuse_wfs_method".')
    if 'soc' in experimental:
        warnings.warn('Please use new "soc" parameter.',
                      DeprecatedParameterWarning)
        assert soc is None
        soc = experimental.pop('soc')
    if 'magmoms' in experimental:
        warnings.warn('Please use new "magmoms" parameter.',
                      DeprecatedParameterWarning)
        assert magmoms is None
        magmoms = experimental.pop('magmoms')
    unknown = experimental.keys() - {'backwards_compatible',
                                     'ccirs',
                                     'pw_pot_calc',
                                     'new_basis'}
    if unknown:
        warnings.warn(f'Unknown experimental keyword(s): {unknown}',
                      stacklevel=3)
    return soc, magmoms


def _fix_legacy_stuff(params: Parameters) -> None:
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


def DFT(
    atoms: Atoms,
    *,
    mode: str | dict | Mode,
    basis: str | dict[str | int | None, str] | None = None,
    charge: float | None = None,
    convergence: dict | None = None,
    eigensolver: str | dict | Eigensolver | None = None,
    experimental: dict | None = None,
    extensions: Sequence[ExtensionInput] | None = None,
    gpts: Sequence[int] | None = None,
    h: float | None = None,
    hund: bool | None = None,
    interpolation: int | None = None,
    kpts: KptsType | MonkhorstPack | None = None,
    magmoms: Sequence[float] | Sequence[Sequence[float]] | None = None,
    maxiter: int | None = None,
    mixer: dict | Mixer | None = None,
    nbands: int | str | None = None,
    occupations: dict | Occupations | None = None,
    parallel: dict | None = None,
    poissonsolver: dict | PoissonSolver | None = None,
    random: bool | None = None,
    setups: str | dict | None = None,
    soc: bool | None = None,
    spinpol: bool | None = None,
    symmetry: str | dict | Symmetry | None = None,
    xc: str | dict | XC | None = None,
    txt: str | Path | IO[str] | None = '-',
    communicator: MPIComm | None = None) -> DFTCalculation:
    """Create a DFTCalculation object.

    See :class:`gpaw.dft.Parameters` for the complete list of parameters.

    Parameters
    ==========
    atoms:
        ASE-Atoms object.
    txt:
        Text log-file.  Use ``None`` for no logging and ``'-'`` for using
        standard out.
    communicator:
        MPI-communicator.  Default is to use ``gpaw.mpi.world``.

    """
    params = Parameters(**{k: v for k, v in locals().items()
                           if k in PARAMETER_NAMES})
    return params.dft_calculation(atoms, txt, communicator)


class LegacyGPAWError(Exception):
    """Something not quite working with new GPAW - try old ..."""


def GPAW(
    filename: str | Path | IO[str] | None = None,
    *,
    basis: str | dict[str | int | None, str] | None = None,
    charge: float | None = None,
    convergence: dict | None = None,
    eigensolver: str | dict | Eigensolver | None = None,
    experimental: dict | None = None,
    extensions: Sequence[ExtensionInput] | None = None,
    gpts: Sequence[int] | None = None,
    h: float | None = None,
    hund: bool | None = None,
    interpolation: int | None = None,
    kpts: KptsType | MonkhorstPack | None = None,
    magmoms: Sequence[float] | Sequence[Sequence[float]] | None = None,
    maxiter: int | None = None,
    mixer: dict | Mixer | None = None,
    mode: str | dict | Mode | None = None,
    nbands: int | str | None = None,
    occupations: dict | Occupations | None = None,
    parallel: dict | None = None,
    poissonsolver: dict | PoissonSolver | None = None,
    random: bool | None = None,
    setups: str | dict | None = None,
    soc: bool | None = None,
    spinpol: bool | None = None,
    symmetry: str | dict | Symmetry | None = None,
    xc: str | dict | XC | None = None,
    txt: str | Path | IO[str] | None = '?',
    communicator: MPIComm | None = None,
    object_hooks=None,
    legacy_gpaw: bool | None = None,
    external=None,
    background_charge=None) -> ASECalculator:
    """Create ASE-compatible GPAW calculator.

    See :class:`gpaw.dft.Parameters` for the complete list of parameters.

    Parameters
    ==========
    filename:
        Name of gpw-file to restart from.
    txt:
        Text log-file.  Use ``None`` for no logging and ``'-'``
        for using standard out.
    communicator:
        MPI-communicator.  Default is to use ``gpaw.mpi.world``.
    object_hooks:
        Dictionary of hook-functions to create custom parameter-objects.
    """
    from gpaw.new.ase_interface import ASECalculator
    from gpaw.new.gpw import read_gpw

    if mode is None:
        del mode

    kwargs = {key: value for key, value in locals().items()
              if key in PARAMETER_NAMES}
    for key in ['background_charge', 'external']:
        value = kwargs[key]
        if value is None:
            del kwargs[key]
        else:
            legacy_gpaw = True

    # Sorry about the following mess, but it will become a lot simpler
    # in the near future!
    params = None
    _use_old_if_reading_new_fails = False
    if legacy_gpaw is None:
        if GPAW_NEW == 147:
            can, params = _can_use_new(filename, kwargs)
            legacy_gpaw = not can
            _use_old_if_reading_new_fails = True
        else:
            legacy_gpaw = False

    if legacy_gpaw:
        from gpaw.old.calculator import GPAW as OldGPAW
        kwargs = {key: value
                  for key, value in kwargs.items() if value is not None}
        return OldGPAW(filename,
                       txt=txt,
                       communicator=communicator,
                       **kwargs)

    if txt == '?':
        txt = '-' if filename is None else None

    log = Logger(txt, communicator)

    if filename is not None:
        args = Parameters(mode='pw', **kwargs)._non_defaults
        if set(args) > {'mode', 'parallel'}:
            raise ValueError(
                'Illegal argument(s) when reading from a file: '
                f'{", ".join(args)}')

        try:
            atoms, dft, _ = read_gpw(filename,
                                     log=log,
                                     parallel=parallel,
                                     object_hooks=object_hooks)
        except LegacyGPAWError:
            if not _use_old_if_reading_new_fails:
                raise
            return GPAW(filename,
                        legacy_gpaw=True,
                        txt=txt,
                        communicator=communicator)

        return ASECalculator(dft.params,
                             log=log, dft=dft, atoms=atoms)

    params = params or Parameters(**kwargs)
    return ASECalculator(params, log=log)


def _can_use_new(filename, kwargs) -> tuple[bool, Parameters | None]:
    """Decide if the parameters are compatible with new-GPAW."""
    if filename is not None:
        return True, None

    try:
        params = Parameters(**kwargs)
    except LegacyGPAWError:
        return False, None
    if params.mode.name == 'lcaooooooooooooo':
        return False, None
    xcname = params.xc.name
    if xcname.startswith(('GLLB', 'TB09', 'LCY', 'CAMY')):
        return False, None
    if params.mode.name == 'fd' and xcname in ['EXX', 'PBE0', 'B3LYP']:
        return False, None
    return True, params
