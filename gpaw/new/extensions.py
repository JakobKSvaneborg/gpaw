from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.units import Bohr, Ha

from gpaw.core import PWArray, PWDesc, UGArray
from gpaw.core.domain import Domain
from gpaw.dft import ExtensionInput
from gpaw.mpi import broadcast_exception, broadcast_float, serial_comm
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.poisson import PoissonSolver


class Extension:
    """Extension object.

    Used for jellium, solvation, solvated jellium model, D3, ...
    """

    name = 'unnamed extension'
    charge = 0.0

    def get_energy_contributions(self) -> dict[str, float]:
        return {}

    def force_contribution(self, nt_r, vHt_r):
        return 0.0

    def stress_contribution(self):
        return 0.0

    def move_atoms(self, relpos_ac) -> None:
        return

    def update_non_local_hamiltonian(self,
                                     D_sii,
                                     setup,
                                     atom_index,
                                     dH_sii) -> float:
        return 0.0

    def build(self, builder):
        return self

    def create_poisson_solver(self,
                              grid,
                              pw,
                              *,
                              charge,
                              xp) -> PoissonSolver | None:
        return None

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             nelectrons: float,
                             occ_calc,
                             mixer,
                             log) -> bool:
        """Allow for environment to "converge"."""
        return True

    def update1(self, nt_r) -> None:
        """Hook called right before solving the Poisson equation."""
        pass

    def update1pw(self, nt_g) -> None:
        """PW-mode hook called right before solving the Poisson equation."""
        pass

    def update2(self, nt_r, vHt_r, vt_sr) -> float:
        """Calculate environment energy."""
        return 0.0

    def update_potential(self,
                         vt_sR: UGArray,
                         density) -> float:
        return 0.0


class D3(ExtensionInput):
    name = 'd3'

    def __init__(self, *, xc, **kwargs):
        self.xc = xc
        self.kwargs = kwargs

    def todict(self) -> dict:
        return {'xc': self.xc, **self.kwargs}

    def build(self, builder: DFTComponentsBuilder):
        from ase.calculators.dftd3 import PureDFTD3
        atoms = builder.atoms.copy()
        world = builder.communicators['w']
        log = builder.log

        # Since DFTD3 is filesystem based, and GPAW has no such requirements
        # we need to be absolutely sure that there are no race-conditions
        # in files. label cannot be used, because dftd3 executable still
        # writes gradients to fixed files, thus a unique folder needs to be
        # created.

        class D3Extension(Extension):
            name = 'd3'

            def __init__(self):
                super().__init__()
                self.stress_vv = np.zeros((3, 3)) * np.nan
                self.F_av = np.zeros_like(atoms.positions) * np.nan
                self.E = np.nan
                self._calculate(atoms)

            def _calculate(_self, atoms):
                # Circumvent a DFTD3 bug for an isolated atom ASE #1672
                if len(atoms) == 1 and not atoms.pbc.any():
                    _self.stress_vv = np.zeros((3, 3)) * np.nan
                    _self.F_av = np.zeros_like(atoms.positions)
                    _self.E = 0.0
                    return

                cwd = Path.cwd()
                assert atoms.calc is None
                # Call DFTD3 only with single core due to #1671
                with broadcast_exception(world):
                    if world.rank == 0:
                        try:
                            _self.calculate_single_core()
                        finally:
                            os.chdir(cwd)
                _self.E = broadcast_float(_self.E, world)
                world.broadcast(_self.F_av, 0)
                world.broadcast(_self.stress_vv, 0)

            def calculate_single_core(_self):
                """Single core method to calculate D3 forces and stresses"""

                label = uuid.uuid4().hex[:8]
                directory = Path('dftd3-ext-' + label).absolute()
                directory.mkdir()

                # Due to ase #1673, relative folders are not supported
                # neither are absolute folders due to 80 character limit.
                # The only way out, is to chdir to a temporary folder here.
                os.chdir(directory)
                log('Evaluating D3 corrections at temporary'
                    f' folder {directory}')
                atoms.calc = PureDFTD3(xc=self.xc,
                                       directory='.',
                                       comm=serial_comm,
                                       **self.kwargs)

                # XXX params.xc should be taken directly from the calculator.
                # XXX What if this is changed via set?
                _self.F_av = atoms.get_forces() / Ha * Bohr

                try:
                    # Copy needed because array is not c-contigous
                    _self.stress_vv = atoms.get_stress(voigt=False).copy() \
                        / Ha * Bohr**3
                except PropertyNotImplementedError:
                    _self.stress_vv = np.zeros((3, 3)) * np.nan

                _self.E = atoms.get_potential_energy() / Ha
                try:
                    os.unlink('ase_dftd3.out')
                    os.unlink('ase_dftd3.POSCAR')
                    os.unlink('dftd3_cellgradient')
                    os.unlink('dftd3_gradient')
                    os.rmdir(directory.absolute())
                except OSError as e:
                    log('Unable to remove files and folder', e)
                atoms.calc = None

            def get_energy_contributions(_self) -> dict[str, float]:
                """Returns the energy contributions from D3 in Hartree"""
                return {f'D3 (xc={self.xc})': _self.E}

            def get_energy(self) -> float:
                """Returns the energy contribution from D3 in eV"""
                return self.E * Ha

            def force_contribution(self, nt_r, vHt_r):
                return self.F_av

            def stress_contribution(self):
                if np.isnan(self.stress_vv).all():
                    raise PropertyNotImplementedError
                return self.stress_vv

            def move_atoms(self, relpos_ac) -> None:
                atoms.set_scaled_positions(relpos_ac)
                self._calculate(atoms)

        return D3Extension()


class Jellium(ExtensionInput):
    name = 'jellium'

    def __init__(self,
                 charge: float):
        self.charge = charge

    def todict(self):
        return {'charge': self.charge}

    def update_mask(self, mask_r) -> None:
        mask_r.data[:] = 1.0

    def build(self, builder: DFTComponentsBuilder):
        mask_r = builder.fine_grid.zeros()
        self.update_mask(mask_r)
        # PW-mode needs this one:
        # pw = builder.electrostatic_potential_desc
        pw = builder.interpolation_desc
        return JelliumExtension(mask_r, charge=self.charge, pw=pw)


class JelliumExtension(Extension):
    def __init__(self,
                 mask_r: UGArray,
                 *,
                 charge: float,
                 pw: Domain):
        self.charge = charge
        self.mask_r = mask_r
        mask_r.data *= 1.0 / mask_r.integrate()
        self.mask_g = None
        if isinstance(pw, PWDesc):
            mask_r = mask_r.gather()
            if mask_r is not None:
                self.mask_g = mask_r.fft(pw=pw.new(comm=None))

    def update1(self, nt_r: UGArray) -> None:
        nt_r.data -= self.mask_r.data * self.charge

    def update1pw(self, nt_g: PWArray | None) -> None:
        if nt_g is None:
            return  # only rank-0 needs to do anything
        assert self.mask_g is not None
        nt_g.data -= self.mask_g.data * self.charge


class FixedPotentialJelliumExtension(JelliumExtension):
    def __init__(self,
                 mask_r: UGArray,
                 *,
                 pw: Domain,
                 workfunction_target: float,  # eV
                 excess_electrons_guess=0.0,
                 tolerance: float = 0.001):  # eV
        """Adjust jellium charge to get the desired Fermi-level."""
        super().__init__(mask_r, charge=excess_electrons_guess or 0.0, pw=pw)
        self.workfunction_target = workfunction_target / Ha
        self.tolerance = tolerance / Ha
        # (Charge, Fermi-level) history:
        self.history: list[tuple[float, float]] = []

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             nelectrons: float,
                             occ_calc,
                             mixer,
                             log) -> bool:
        fl1 = ibzwfs.fermi_level
        log(f'charge: {self.charge:.6f} |e|, Fermi-level: {fl1 * Ha:.3f} eV')
        fl = -self.workfunction_target
        if abs(fl1 - fl) <= self.tolerance:
            return True
        self.history.append((self.charge, fl1))
        if len(self.history) == 1:
            area = abs(np.linalg.det(self.mask_r.desc.cell_cv[:2, :2]))
            dc = -(fl1 - fl) * area * 0.02
        else:
            (c2, fl2), (c1, fl1) = self.history[-2:]
            c = c2 + (fl - fl2) / (fl1 - fl2) * (c1 - c2)
            dc = c - c1
            if abs(dc) > abs(c2 - c1):
                dc *= abs((c2 - c1) / dc)
        self.charge += dc
        nelectrons += dc
        ibzwfs.calculate_occs(occ_calc, nelectrons)
        mixer.reset()
        return False
