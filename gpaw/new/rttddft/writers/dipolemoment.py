import json

from ase.utils import IOContext

from gpaw.mpi import world, MPIComm
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.rttddft.history import RTTDDFTHistory, RTTDDFTKick
from gpaw.new.rttddft.state import RTTDDFTState


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
        line += '# %20s %22s %22s %22s\n' % ('time', 'dmx', 'dmy', 'dmz')
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

    def _write_kick(self,
                    kick: RTTDDFTKick):
        comment = f'Kick = {json.dumps(kick.todict())}'
        self.write_comment(comment)

    def write_dm(self,
                 history: RTTDDFTHistory,
                 state: RTTDDFTState,
                 pot_calc: PotentialCalculator):
        """ Calculate the dipole moment from the state and write to file.

        If one or several kicks were just performed, then also a comment
        is written.

        Parameters
        ----------
        history
            RTTDDFT history object.
        state
            State containing wave functions and potentials.
        pot_calc
            Potential calculator.
        """
        relpos_ac = pot_calc.relpos_ac
        dipolemoment = state.density.calculate_dipole_moment(relpos_ac)

        # Write any kick that was just performed
        for kick in history.kicks:
            if kick.time < history.time:
                # Skip kicks at previous times
                continue
            self._write_kick(kick)

        data = (history.time, ) + tuple(dipolemoment)
        line = '%20.8lf %22.12le %22.12le %22.12le\n' % data

        self._write(line)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.ioctx.close()

    def __del__(self):
        self.ioctx.close()
