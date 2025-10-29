from __future__ import annotations

from typing import IO, Any, Union
from pathlib import Path

from ase import Atoms
import ase.io.ulm as ulm
from ase.io.trajectory import read_atoms, write_atoms
from ase.units import Bohr, Ha
import gpaw
import gpaw.mpi as mpi
from gpaw.dft import Parameters
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.gpw import GPWFlags, write_dft_state, read_dft_state
from gpaw.new.rttddft.history import RTTDDFTHistory
from gpaw.new.rttddft.dataclasses import RTTDDFTState
from gpaw.new.logger import Logger


def write_rttddft(filename: str | Path,
                  atoms: Atoms,
                  dft_params: Parameters,
                  td_params: dict[str, Any],
                  state: RTTDDFTState,
                  history: RTTDDFTHistory) -> None:
    flags = GPWFlags(include_wfs=True, include_projections=True,
                     precision='double')
    comm = mpi.world

    writer: ulm.Writer | ulm.DummyWriter
    if comm.rank == 0:
        writer = ulm.Writer(filename, tag='gpaw-rttddft')
    else:
        writer = ulm.DummyWriter()

    with writer:
        writer.write(version=0,
                     gpaw_version=gpaw.__version__,
                     ha=Ha,
                     bohr=Bohr,
                     precision=flags.precision)

        write_atoms(writer.child('atoms'), atoms)

        writer.child('parameters').write(**td_params)

        statewriter = writer.child('state')
        statewriter.write(version=6,  # Corresponding to DFT version
                          ha=Ha,
                          bohr=Bohr)
        statewriter.child('parameters').write(**dft_params.todict())
        write_dft_state(statewriter, dft_params,
                        ibzwfs=state.ibzwfs,
                        density=state.density,
                        potential=state.potential,
                        energies=state.energies,
                        flags=flags)

        historywriter = writer.child('history')
        historywriter.write(**history.todict())

    comm.barrier()


def read_rttddft(filename: Union[str, Path, IO[str]],
                 *,
                 log: Union[Logger, str, Path, IO[str]] = None,
                 comm=None,
                 parallel: dict[str, Any] = None,
                 ) -> tuple[Atoms,
                            RTTDDFTState,
                            RTTDDFTHistory,
                            Parameters,
                            dict[str, Any],
                            DFTComponentsBuilder]:
    """
    Read RTTDDFT file
    """

    parallel = parallel or {}

    if not isinstance(log, Logger):
        log = Logger(log, comm or mpi.world)

    comm = log.comm

    log(f'Reading from {filename}')

    reader = ulm.Reader(filename)

    atoms = read_atoms(reader.atoms)
    td_params = reader.parameters.asdict()

    # Read state parameters
    kwargs = reader.state.parameters.asdict()
    kwargs['parallel'] = parallel
    dft_params = Parameters(**kwargs)

    # In RTTDDFT we always force complex dtype and disable symmetries
    dft_params.mode.force_complex_dtype = True
    dft_params.symmetry.point_group = False

    # Read state arrays and create the builder
    builder, dft_params, dft_state = read_dft_state(
        reader.state, atoms=atoms, params=dft_params,
        comm=comm, singlep=False, log=log)
    ibzwfs, density, potential, energies = dft_state
    state = RTTDDFTState(*dft_state)

    if builder.mode in ['pw', 'fd']:  # fd = finite-difference
        data = ibzwfs.wfs_qs[0][0].psit_nX.data
        if not hasattr(data, 'fd'):  # fd = file-descriptor
            reader.close()
    else:
        reader.close()

    history = RTTDDFTHistory.from_values(**reader.history.asdict())

    return atoms, state, history, dft_params, td_params, builder
