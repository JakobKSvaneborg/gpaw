import json

from ase.utils import IOContext

from gpaw.mpi import world, MPIComm
from gpaw.external import ExternalPotential
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.rttddft.dataclasses import RTTDDFTState, RTTDDFTKick


class DipoleMomentWriter:
    """Observer for writing time-dependent dipole moment data.

    The data is written in atomic units.

    Parameters
    ----------
    filename
        File for writing dipole moment data.
    append
        Append to file if True, otherwise create new file (erasing any
        existing one).
    comm
        MPI communicator corresponding to world. By default,
        the actual world is used.
    """
    version = 2

    def __init__(self,
                 filename: str, *,
                 append: bool = False,
                 comm: MPIComm | None = None):
        if comm is None:
            comm = world

        self.ioctx = IOContext()

        if append:
            # Open in append mode
            self.fd = self.ioctx.openfile(filename, comm=comm, mode='a')
        else:
            # Create new file and write header
            self.fd = self.ioctx.openfile(filename, comm=comm, mode='w')
            self._write_header()

    def _write(self, line):
        self.fd.write(line)
        self.fd.flush()

    def _write_header(self):
        line = f'# {self.__class__.__name__}[version={self.version}]\n'
        line += '# Using Hartree atomic units for time and dipole moment\n'
        line += '# %18s %22s %22s %22s\n' % ('time', 'dmx', 'dmy', 'dmz')
        self._write(line)

    def write_comment(self,
                      comment: str):
        """ Write comment to the dipole moment file.

        Parameters
        ----------
        comment
            Comment string. A comment character (#) is prepended to
            every line of the comment and trailing newline is added.
        """
        lines = ['# ' + line for line in comment.split('\n')]
        lines.append('')  # Add one traling newline
        line = '\n'.join(lines)
        self._write(line)

    def write_kick(self,
                   time: float,
                   potential: ExternalPotential,
                   gauge: str = 'length'):
        """ Write a comment with a description of the kick.

        This comment is formatted such that it can be parsed by the
        spectrum calculator.

        Parameters
        ----------
        time
            Current simulation time in atomic units.
        potential
            The external potential of the kick.
        gauge
            Kick gauge.
        """
        kick = RTTDDFTKick(time=time, potential=potential, gauge=gauge)
        comment = f'Kick = {json.dumps(kick.todict())}'
        self.write_comment(comment)

    def write_dm(self,
                 time: float,
                 state: RTTDDFTState,
                 pot_calc: PotentialCalculator):
        """ Calculate the dipole moment from the state and write to file.

        Parameters
        ----------
        time
            Current simulation time in atomic units.
        state
            State containing wave functions and potentials.
        pot_calc
            Potential calculator.
        """
        relpos_ac = pot_calc.relpos_ac
        dipolemoment = state.density.calculate_dipole_moment(relpos_ac)

        data = (time, ) + tuple(dipolemoment)
        line = '%20.8lf %22.12le %22.12le %22.12le\n' % data

        self._write(line)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.ioctx.close()

    def __del__(self):
        self.ioctx.close()
