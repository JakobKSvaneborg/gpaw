from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, Union

import numpy as np
from ase import Atoms
from ase.units import Bohr, Ha
from gpaw import __version__
from gpaw.core import UGArray
from gpaw.dos import DOSCalculator
from gpaw.mpi import world, synchronize_atoms, broadcast as bcast
from gpaw.new import Timer
from gpaw.new.builder import builder as create_builder
from gpaw.new.calculation import (DFTCalculation, DFTState,
                                  ReuseWaveFunctionsError, units)
from gpaw.new.gpw import read_gpw, write_gpw
from gpaw.new.input_parameters import (DeprecatedParameterWarning,
                                       InputParameters)
from gpaw.new.logger import Logger
from gpaw.new.pw.fulldiag import diagonalize
from gpaw.new.xc import create_functional
from gpaw.typing import Array1D, Array2D, Array3D
from gpaw.utilities import pack
from gpaw.utilities.memory import maxrss


def GPAW(filename: Union[str, Path, IO[str]] = None,
         txt: str | Path | IO[str] | None = '?',
         communicator=None,
         **kwargs) -> ASECalculator:
    """Create ASE-compatible GPAW calculator."""
    if txt == '?':
        txt = '-' if filename is None else None

    parallel = kwargs.get('parallel', {})
    comm = parallel.pop('world', None)
    if comm is None:
        comm = communicator or world
    else:
        warnings.warn(('Please use communicator=... '
                       'instead of parallel={''world'': ...}'),
                      DeprecatedParameterWarning)
    log = Logger(txt, comm)

    if filename is not None:
        if not {'parallel'}.issuperset(kwargs):
            illegal = set(kwargs) - {'parallel'}
            raise ValueError('Illegal arguments when reading from a file: '
                             f'{illegal}')
        atoms, calculation, params, _ = read_gpw(filename,
                                                 log=log,
                                                 parallel=parallel)
        return ASECalculator(params,
                             log=log, calculation=calculation, atoms=atoms)

    params = InputParameters(kwargs)
    write_header(log, params)
    return ASECalculator(params, log=log)


def write_header(log, params):
    from gpaw.io.logger import write_header as header
    log(f'#  __  _  _\n# | _ |_)|_||  |\n# |__||  | ||/\\| - {__version__}\n')
    header(log, log.comm)
    log('---')
    with log.indent('input parameters:'):
        log(**dict(params.items()))


def compare_atoms(a1: Atoms, a2: Atoms) -> set[str]:
    if a1 is a2:
        return set()

    if len(a1.numbers) != len(a2.numbers) or (a1.numbers != a2.numbers).any():
        return {'numbers'}

    if (a1.pbc != a2.pbc).any():
        return {'pbc'}

    if abs(a1.cell - a2.cell).max() > 0.0:
        return {'cell'}

    if abs(a1.positions - a2.positions).max() > 0.0:
        return {'positions'}

    return set()


class ASECalculator:
    """This is the ASE-calculator frontend for doing a GPAW calculation."""

    name = 'gpaw'

    def __init__(self,
                 params: InputParameters,
                 *,
                 log: Logger,
                 calculation=None,
                 atoms=None):
        self.params = params
        self.log = log
        self.comm = log.comm
        self.calculation = calculation

        self.atoms = atoms
        self.timer = Timer()

    def __repr__(self):
        params = []
        for key, value in self.params.items():
            val = repr(value)
            if len(val) > 40:
                val = '...'
            params.append((key, val))
        p = ', '.join(f'{key}: {val}' for key, val in params)
        return f'ASECalculator({p})'

    def calculate_property(self,
                           atoms: Atoms | None,
                           prop: str) -> Any:
        """Calculate (if not already calculated) a property.

        The ``prop`` string must be one of

        * energy
        * forces
        * stress
        * magmom
        * magmoms
        * dipole
        """
        if atoms is None:
            atoms = self.atoms
        else:
            synchronize_atoms(atoms, self.comm)
        assert atoms is not None

        if self.calculation is not None:
            changes = compare_atoms(self.atoms, atoms)
            if changes & {'numbers', 'pbc', 'cell'}:
                if 'numbers' not in changes:
                    # Remember magmoms if there are any:
                    magmom_a = self.calculation.results.get('magmoms')
                    if magmom_a is not None and magmom_a.any():
                        atoms = atoms.copy()
                        assert atoms is not None  # MYPY: why is this needed?
                        atoms.set_initial_magnetic_moments(magmom_a)

                if changes & {'numbers', 'pbc'}:
                    self.calculation = None  # start from scratch
                else:
                    try:
                        self.create_new_calculation_from_old(atoms)
                    except ReuseWaveFunctionsError:
                        self.calculation = None  # start from scratch
                    else:
                        self.converge()
                        changes = set()

        if self.calculation is None:
            self.create_new_calculation(atoms)
            assert self.calculation is not None
            self.converge()
        elif changes:
            self.move_atoms(atoms)
            self.converge()

        if prop == 'forces':
            with self.timer('Forces'):
                self.calculation.forces()
        elif prop == 'stress':
            with self.timer('Stress'):
                self.calculation.stress()
        elif prop == 'dipole':
            self.calculation.dipole()
        elif prop not in self.calculation.results:
            raise KeyError('Unknown property:', prop)

        return self.calculation.results[prop] * units[prop]

    def get_property(self,
                     name: str,
                     atoms: Atoms | None = None,
                     allow_calculation: bool = True) -> Any:
        if not allow_calculation and name not in self.calculation.results:
            return None
        if atoms is None:
            atoms = self.atoms
        return self.calculate_property(atoms, name)

    @property
    def results(self):
        if self.calculation is None:
            return {}
        return {name: value * units[name]
                for name, value in self.calculation.results.items()}

    def create_new_calculation(self, atoms: Atoms) -> None:
        with self.timer('Init'):
            self.calculation = DFTCalculation.from_parameters(
                atoms, self.params, self.comm, self.log)
        self.atoms = atoms.copy()

    def create_new_calculation_from_old(self, atoms: Atoms) -> None:
        with self.timer('Morph'):
            self.calculation = self.calculation.new(
                atoms, self.params, self.log)
        self.atoms = atoms.copy()

    def move_atoms(self, atoms):
        with self.timer('Move'):
            self.calculation = self.calculation.move_atoms(atoms)
        self.atoms = atoms.copy()

    def converge(self):
        """Iterate to self-consistent solution.

        Will also calculate "cheap" properties: energy, magnetic moments
        and dipole moment.
        """
        with self.timer('SCF'):
            self.calculation.converge(calculate_forces=self._calculate_forces)

        # Calculate all the cheap things:
        self.calculation.energies()
        self.calculation.dipole()
        self.calculation.magmoms()

        self.calculation.write_converged()

    def _calculate_forces(self) -> Array2D:  # units: Ha/Bohr
        """Helper method for force-convergence criterium."""
        with self.timer('Forces'):
            self.calculation.forces(silent=True)
        return self.calculation.results['forces'].copy()

    def __del__(self):
        self.log('---')
        self.timer.write(self.log)
        try:
            mib = maxrss() / 1024**2
            self.log(f'\nMax RSS: {mib:.3f}  # MiB')
        except NameError:
            pass

    def get_potential_energy(self,
                             atoms: Atoms | None = None,
                             force_consistent: bool = False) -> float:
        return self.calculate_property(atoms,
                                       'free_energy' if force_consistent else
                                       'energy')

    def get_forces(self, atoms: Atoms | None = None) -> Array2D:
        return self.calculate_property(atoms, 'forces')

    def get_stress(self, atoms: Atoms | None = None) -> Array1D:
        return self.calculate_property(atoms, 'stress')

    def get_dipole_moment(self, atoms: Atoms | None = None) -> Array1D:
        return self.calculate_property(atoms, 'dipole')

    def get_magnetic_moment(self, atoms: Atoms | None = None) -> float:
        return self.calculate_property(atoms, 'magmom')

    def get_magnetic_moments(self, atoms: Atoms | None = None) -> Array1D:
        return self.calculate_property(atoms, 'magmoms')

    def get_non_collinear_magnetic_moment(self,
                                          atoms: Atoms | None = None
                                          ) -> Array1D:
        return self.calculate_property(atoms, 'non_collinear_magmom')

    def get_non_collinear_magnetic_moments(self,
                                           atoms: Atoms | None = None
                                           ) -> Array2D:
        return self.calculate_property(atoms, 'non_collinear_magmoms')

    def write(self, filename, mode=''):
        """Write calculator object to a file.

        Parameters
        ----------
        filename:
            File to be written
        mode:
            Write mode. Use ``mode='all'``
            to include wave functions in the file.
        """
        self.log(f'# Writing to {filename} (mode={mode!r})\n')

        write_gpw(filename, self.atoms, self.params,
                  self.calculation, skip_wfs=mode != 'all')

    # Old API:

    implemented_properties = ['energy', 'free_energy',
                              'forces', 'stress',
                              'dipole', 'magmom', 'magmoms']

    def new(self, **kwargs) -> ASECalculator:
        kwargs = {**dict(self.params.items()), **kwargs}
        return GPAW(**kwargs)

    def get_pseudo_wave_function(self, band, kpt=0, spin=None,
                                 periodic=False,
                                 broadcast=True) -> Array3D:
        state = self.calculation.state
        collinear = state.ibzwfs.collinear
        if collinear:
            if spin is None:
                spin = 0
        else:
            assert spin is None
        wfs = state.ibzwfs.get_wfs(spin=spin if collinear else 0,
                                   kpt=kpt,
                                   n1=band, n2=band + 1)
        if wfs is not None:
            basis = getattr(self.calculation.scf_loop.hamiltonian,
                            'basis', None)
            grid = state.density.nt_sR.desc.new(comm=None)
            if collinear:
                wfs = wfs.to_uniform_grid_wave_functions(grid, basis)
                psit_R = wfs.psit_nX[0]
            else:
                psit_sG = wfs.psit_nX[0]
                grid = grid.new(kpt=psit_sG.desc.kpt_c,
                                dtype=psit_sG.desc.dtype)
                psit_R = psit_sG.ifft(grid=grid)
            if not psit_R.desc.pbc.all():
                psit_R = psit_R.to_pbc_grid()
            if periodic:
                psit_R.multiply_by_eikr(-psit_R.desc.kpt_c)
            array_R = psit_R.data * Bohr**-1.5
        else:
            array_R = None
        if broadcast:
            array_R = bcast(array_R, 0, self.calculation.comm)
        return array_R

    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    def get_fermi_level(self) -> float:
        state = self.calculation.state
        fl = state.ibzwfs.fermi_levels * Ha
        assert len(fl) == 1
        return fl[0]

    def get_fermi_levels(self) -> float:
        state = self.calculation.state
        fl = state.ibzwfs.fermi_levels * Ha
        assert len(fl) == 2
        return fl

    def get_homo_lumo(self, spin: int = None) -> Array1D:
        state = self.calculation.state
        return state.ibzwfs.get_homo_lumo(spin) * Ha

    def get_number_of_electrons(self):
        state = self.calculation.state
        return state.ibzwfs.nelectrons

    def get_number_of_bands(self):
        state = self.calculation.state
        return state.ibzwfs.nbands

    def get_number_of_grid_points(self):
        return self.calculation.state.density.nt_sR.desc.size

    def get_effective_potential(self, spin=0):
        assert spin == 0
        vt_R = self.calculation.state.potential.vt_sR[spin]
        return vt_R.to_pbc_grid().gather(broadcast=True).data * Ha

    def get_electrostatic_potential(self):
        density = self.calculation.state.density
        potential, _ = self.calculation.pot_calc.calculate(density)
        vHt_x = potential.vHt_x
        if isinstance(vHt_x, UGArray):
            return vHt_x.gather(broadcast=True).to_pbc_grid().data * Ha

        grid = self.calculation.pot_calc.fine_grid
        return vHt_x.ifft(grid=grid).gather(broadcast=True).data * Ha

    def get_atomic_electrostatic_potentials(self):
        return self.calculation.electrostatic_potential().atomic_potentials()

    def get_electrostatic_corrections(self):
        return self.calculation.electrostatic_potential().atomic_corrections()

    def get_pseudo_density(self,
                           spin=None,
                           gridrefinement=1,
                           broadcast=True) -> Array3D:
        assert spin is None
        nt_sr = self.calculation.densities().pseudo_densities(
            grid_refinement=gridrefinement)
        return nt_sr.gather(broadcast=broadcast).data.sum(0)

    def get_all_electron_density(self,
                                 spin=None,
                                 gridrefinement=1,
                                 broadcast=True,
                                 skip_core=False):
        assert spin is None
        n_sr = self.calculation.densities().all_electron_densities(
            grid_refinement=gridrefinement,
            skip_core=skip_core)
        return n_sr.gather(broadcast=broadcast).data.sum(0)

    def get_eigenvalues(self, kpt=0, spin=0, broadcast=True):
        state = self.calculation.state
        eig_n = state.ibzwfs.get_eigs_and_occs(k=kpt, s=spin)[0] * Ha
        if broadcast:
            if self.comm.rank != 0:
                eig_n = np.empty(state.ibzwfs.nbands)
            self.comm.broadcast(eig_n, 0)
        return eig_n

    def get_occupation_numbers(self, kpt=0, spin=0, broadcast=True):
        state = self.calculation.state
        weight = state.ibzwfs.ibz.weight_k[kpt] * state.ibzwfs.spin_degeneracy
        occ_n = state.ibzwfs.get_eigs_and_occs(k=kpt, s=spin)[1] * weight
        if broadcast:
            if self.comm.rank != 0:
                occ_n = np.empty(state.ibzwfs.nbands)
            self.comm.broadcast(occ_n, 0)
        return occ_n

    def get_reference_energy(self):
        return self.calculation.setups.Eref * Ha

    def get_number_of_iterations(self):
        return self.calculation.scf_loop.niter

    def get_bz_k_points(self):
        state = self.calculation.state
        return state.ibzwfs.ibz.bz.kpt_Kc.copy()

    def get_ibz_k_points(self):
        state = self.calculation.state
        return state.ibzwfs.ibz.kpt_kc.copy()

    def get_orbital_magnetic_moments(self):
        """Return the orbital magnetic moment vector for each atom."""
        from gpaw.new.orbmag import get_orbmag_from_calc
        return get_orbmag_from_calc(self)

    def calculate(self, atoms, properties=None, system_changes=None):
        if properties is None:
            properties = ['energy']

        for name in properties:
            self.calculate_property(atoms, name)
        # self.get_potential_energy(atoms)

    @cached_property
    def wfs(self):
        from gpaw.new.backwards_compatibility import FakeWFS
        return FakeWFS(self.calculation, self.atoms)

    @property
    def density(self):
        from gpaw.new.backwards_compatibility import FakeDensity
        return FakeDensity(self.calculation)

    @property
    def hamiltonian(self):
        from gpaw.new.backwards_compatibility import FakeHamiltonian
        return FakeHamiltonian(self.calculation)

    @property
    def spos_ac(self):
        return self.atoms.get_scaled_positions()

    @property
    def world(self):
        return self.comm

    @property
    def setups(self):
        return self.calculation.setups

    @property
    def initialized(self):
        return self.calculation is not None

    def get_xc_functional(self):
        return self.calculation.pot_calc.xc.name

    def get_xc_difference(self, xcparams):
        """Calculate non-selfconsistent XC-energy difference."""
        dft = self.calculation
        pot_calc = dft.pot_calc
        state = dft.state
        density = dft.state.density
        xc = create_functional(xcparams, pot_calc.fine_grid)
        if xc.type == 'MGGA' and density.taut_sR is None:
            state.ibzwfs.make_sure_wfs_are_read_from_gpw_file()
            if isinstance(state.ibzwfs.wfs_qs[0][0].psit_nX, SimpleNamespace):
                params = InputParameters(dict(self.params.items()))
                builder = create_builder(self.atoms, params, self.comm)
                basis_set = builder.create_basis_set()
                ibzwfs = builder.create_ibz_wave_functions(
                    basis_set, state.potential, log=dft.log)
                ibzwfs.fermi_levels = state.ibzwfs.fermi_levels
                state.ibzwfs = ibzwfs
                dft.scf_loop.update_density_and_potential = False
                dft.converge()
            density.update_ked(state.ibzwfs)
        exct = pot_calc.calculate_non_selfconsistent_exc(
            xc, density.nt_sR, density.taut_sR)
        dexc = 0.0
        for a, D_sii in state.density.D_asii.items():
            setup = self.setups[a]
            dexc += xc.calculate_paw_correction(
                setup,
                np.array([pack(D_ii) for D_ii in D_sii.real]))
        dexc = state.ibzwfs.domain_comm.sum_scalar(dexc)
        return (exct + dexc - state.potential.energies['xc']) * Ha

    def diagonalize_full_hamiltonian(self,
                                     nbands: int = None,
                                     scalapack=None,
                                     expert: bool = None) -> None:
        if expert is not None:
            warnings.warn('Ignoring deprecated "expert" argument',
                          DeprecationWarning)
        state = self.calculation.state
        ibzwfs = diagonalize(state.potential,
                             state.ibzwfs,
                             self.calculation.scf_loop.occ_calc,
                             nbands,
                             self.calculation.pot_calc.xc)
        self.calculation.state = DFTState(ibzwfs,
                                          state.density,
                                          state.potential)
        nbands = ibzwfs.nbands
        self.params.nbands = nbands
        self.params.keys.append('nbands')

    def gs_adapter(self):
        from gpaw.response.groundstate import ResponseGroundStateAdapter
        return ResponseGroundStateAdapter(self)

    def fixed_density(self, txt='-', **kwargs):
        kwargs = {**dict(self.params.items()), **kwargs}
        params = InputParameters(kwargs)
        log = Logger(txt, self.comm)
        builder = create_builder(self.atoms, params, self.comm)
        basis_set = builder.create_basis_set()
        state = self.calculation.state
        comm1 = state.ibzwfs.kpt_band_comm
        comm2 = builder.communicators['D']
        potential = state.potential.redist(
            builder.grid,
            builder.electrostatic_potential_desc,
            builder.atomdist,
            comm1, comm2)
        density = state.density.redist(builder.grid,
                                       builder.interpolation_desc,
                                       builder.atomdist,
                                       comm1, comm2)
        ibzwfs = builder.create_ibz_wave_functions(basis_set, potential,
                                                   log=log)
        ibzwfs.fermi_levels = state.ibzwfs.fermi_levels
        state = DFTState(ibzwfs, density, potential)
        scf_loop = builder.create_scf_loop()
        scf_loop.update_density_and_potential = False

        calculation = DFTCalculation(
            state,
            builder.setups,
            scf_loop,
            SimpleNamespace(fracpos_ac=self.calculation.fracpos_ac,
                            poisson_solver=None),
            log)

        calculation.converge()

        return ASECalculator(params,
                             log=log,
                             calculation=calculation,
                             atoms=self.atoms)

    def initialize(self, atoms):
        self.create_new_calculation(atoms)

    def converge_wave_functions(self):
        self.calculation.state.ibzwfs.make_sure_wfs_are_read_from_gpw_file()

    def get_number_of_spins(self):
        return self.calculation.state.density.ndensities

    @property
    def parameters(self):
        return self.params

    def dos(self,
            soc: bool = False,
            theta: float = 0.0,  # degrees
            phi: float = 0.0,  # degrees
            shift_fermi_level: bool = True) -> DOSCalculator:
        """Create DOS-calculator.

        Default is to ``shift_fermi_level`` to 0.0 eV.  For ``soc=True``,
        angles can be given in degrees.
        """
        return DOSCalculator.from_calculator(
            self, soc=soc,
            theta=theta, phi=phi,
            shift_fermi_level=shift_fermi_level)

    def band_structure(self):
        """Create band-structure object for plotting."""
        from ase.spectrum.band_structure import get_band_structure
        return get_band_structure(calc=self)

    @property
    def symmetry(self):
        return self.calculation.state.ibzwfs.ibz.symmetries.symmetry
