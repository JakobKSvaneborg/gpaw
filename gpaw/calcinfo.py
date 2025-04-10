from dataclasses import dataclass
from typing import Union

from ase import Atoms

from gpaw.new.builder import builder
from gpaw.new.calculation import DFTCalculation
from gpaw.new.ibzwfs import IBZ
from gpaw.new.logger import Logger
from gpaw.core import UGDesc
from gpaw.core.domain import Domain
from gpaw.setup import Setups
from gpaw.mpi import MPIComm


@dataclass
class CalcInfo:
    atoms: Atoms
    input_params: dict
    ibz: IBZ
    ncomponents: int
    nspins: int
    nbands: int
    nelectrons: float
    setups: Setups
    grid: UGDesc
    wf_description: Union[Domain, None]
    communicators: Union[dict[str, MPIComm], None]
    comm: Union[MPIComm, None]
    log: Union[Logger, str, None]

    def update_params(self, **updated_params):
        params = self.input_params.copy()
        if self.log is not None and 'txt' not in updated_params:
            params['txt'] = self.log
        if self.comm is not None and 'comm' not in updated_params:
            params['comm'] = self.comm
        params.update(updated_params)
        return get_calculation_info(self.atoms, **params)

    def get_dft_calc(self) -> DFTCalculation:
        return DFTCalculation.from_parameters(self.atoms.copy(),
                                              self.input_params,
                                              comm=self.comm,
                                              log=self.log)

    def get_ase_calc(self):
        return self.get_dft_calc().get_ase_calc()


def get_calculation_info(atoms: Atoms,
                         **param_kwargs) -> CalcInfo:
    """
    Get information about a calculation, e.g. grid size, IBZ, nbands,
    parallelization, etc. without actually performing the calculation
    or initializing large arrays.

    Parameters
    ----------
    atoms : Atoms
        Atoms object
    **param_kwargs :
        Input parameters as keyword arguments

    Returns
    -------
    CalcInfo
        Information about the calculation with the given input parameters.

    CalcInfo attributes
    -----
    atoms : Atoms
        Atoms object
    input_params : dict
        Input parameters
    ibz : IBZ
        IBZ object with information about k-point grid
    ncomponents : int
        Number of spin components
    nspins : int
        Number of spin channels
    nbands : int
        Number of bands
    setups : Setups
        Setups object with information about pseudopotentials
    grid : UGDesc
        Grid object with information about the real space grid
    wf_description : Domain
        Domain object with information about the wavefunctions
        (only for non-LCAO calculations)
    communicators : dict
        Dictionary with communicators for k-points, domains and bands
    comm : MPIComm
        MPI communicator
    log : Logger
        Logger object

    CalcInfo methods
    ----------------
    update_params
        Update input parameters and return new CalcInfo object
    get_dft_calc
        Return DFTCalculation object with the given input parameters
    get_ase_calc
        Return ASECalculation object with the given input parameters
    """
    if 'txt' in param_kwargs:
        log = param_kwargs.pop('txt')
    else:
        log = None
    if 'comm' in param_kwargs:
        comm = param_kwargs.pop('comm')
    else:
        comm = None
    dft_builder = builder(atoms, params=param_kwargs, comm=comm, log=log)
    dft_params = CalcInfo(atoms=atoms,
                          input_params=param_kwargs,
                          ibz=dft_builder.ibz,
                          ncomponents=dft_builder.ncomponents,
                          nspins=dft_builder.nspins,
                          nbands=dft_builder.nbands,
                          nelectrons=dft_builder.nelectrons,
                          setups=dft_builder.setups,
                          grid=dft_builder.grid,
                          communicators=dft_builder.communicators,
                          wf_description=dft_builder.create_wf_description()
                          if dft_builder.mode != 'lcao' else None,
                          comm=comm,
                          log=log)
    return dft_params
