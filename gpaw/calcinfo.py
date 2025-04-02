from dataclasses import dataclass
from typing import Union

from ase import Atoms

from gpaw.new.builder import builder
from gpaw.new.calculation import DFTCalculation
from gpaw.new.ibzwfs import IBZ
from gpaw.new.input_parameters import InputParameters
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
    setups: Setups
    grid: UGDesc
    wf_description: Union[Domain, None] = None
    communicators: Union[dict[str, MPIComm], None] = None
    comm: Union[MPIComm, None] = None
    log: Union[Logger, str, None] = None

    def update_params(self, **updated_params):
        params = self.input_params.copy()
        if self.log is not None:
            params['log'] = self.log
        if self.comm is not None:
            params['comm'] = self.comm
        params.update(updated_params)
        return get_calculation_info(self.atoms, params)

    def get_dft_calc(self, updated_params: dict = {},
                     comm=None, log=None) -> DFTCalculation:
        params = self.input_params.copy()
        params.update(updated_params)
        if comm is None:
            comm = self.comm
        if log is None:
            log = self.log
        return DFTCalculation.from_parameters(self.atoms.copy(),
                                              params,
                                              comm=comm,
                                              log=log)

    def get_ase_calc(self, updated_params: dict = {},
                     comm=None, log=None):
        dft = self.get_dft_calc(updated_params, comm, log)
        return dft.get_ase_calc()


def get_calculation_info(atoms: Atoms, *param_dict,
                         **param_kwargs) -> CalcInfo:
    """
    Get information about a calculation, e.g. grid size, IBZ, nbands,
    parallelization, etc. without actually performing the calculation
    or initializing large arrays.

    Parameters
    ----------
    atoms : Atoms
        Atoms object
    param_dict : dict, optional
        Dictionary with input parameters
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
    params = {}
    if len(param_dict) > 1:
        raise TypeError('get_calculation_info got too '
                        'many positional arguments')
    if len(param_dict) == 1:
        if isinstance(param_dict[0], dict):
            params.update(param_dict[0])
        else:
            raise TypeError('get_calculation_info 2nd positional '
                            'argument must be dict if present')
    params.update(param_kwargs)
    if 'log' in params:
        log = params.pop('log')
    else:
        log = None
    if 'comm' in params:
        comm = params.pop('comm')
    else:
        comm = None
    dft_builder = builder(atoms, params=params, comm=comm, log=log)
    dft_params = CalcInfo(atoms,
                          params,
                          dft_builder.ibz,
                          dft_builder.ncomponents,
                          dft_builder.nspins,
                          dft_builder.nbands,
                          dft_builder.setups,
                          dft_builder.grid,
                          dft_builder.communicators,
                          comm=comm,
                          log=log)
    if dft_builder.mode != 'lcao':
        dft_params.wf_description = \
            dft_builder.create_wf_description()
    return dft_params
