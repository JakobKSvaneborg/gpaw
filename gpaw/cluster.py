"""Extensions to the ase Atoms class

"""
import numpy as np

from ase import Atoms
from ase.io import read
from ase.build.connected import connected_indices


class Cluster(Atoms):
    """A class for cluster structures
    to enable simplified manipulation"""

    def __init__(self, *args, **kwargs):

        self.data = {}

        if len(args) > 0:
            filename = args[0]
            if isinstance(filename, str):
                self.read(filename, kwargs.get('filetype'))
                return
        else:
            Atoms.__init__(self, [])

        if kwargs.get('filename') is not None:
            filename = kwargs.pop('filename')
            Atoms.__init__(self, *args, **kwargs)
            self.read(filename, kwargs.get('filetype'))
        else:
            Atoms.__init__(self, *args, **kwargs)

    def extreme_positions(self):
        """get the extreme positions of the structure"""
        pos = self.get_positions()
        return np.array([np.minimum.reduce(pos), np.maximum.reduce(pos)])

    def find_connected(self, index, dmax=None, scale=1.5):
        """Find atoms connected to self[index] and return them."""
        return self[connected_indices(self, index, dmax, scale)]

    def minimal_box(self, border=0, h=None, multiple=4):
        """The box needed to fit the structure in.

        The structure is moved to fit into the box [(0,x),(0,y),(0,z)]
        with x,y,z > 0 (fitting the ASE constriction).
        The border argument can be used to add a border of empty space
        around the structure.

        If h is set, the box is extended to ensure that box/h is
        a multiple of 'multiple'.
        This ensures that GPAW uses the desired h.

        The shift applied to the structure is returned.
         """

        if len(self) == 0:
            return None

        extr = self.extreme_positions()

        # add borders
        if isinstance(border, list):
            b = border
        else:
            b = [border, border, border]
        for c in range(3):
            extr[0][c] -= b[c]
            extr[1][c] += b[c] - extr[0][c]  # shifted already

        # check for multiple of 4
        if h is not None:
            if not hasattr(h, '__len__'):
                h = np.array([h, h, h])
            for c in range(3):
                # apply the same as in paw.py
                L = extr[1][c]  # shifted already
                N = np.ceil(L / h[c] / multiple) * multiple
                # correct L
                dL = N * h[c] - L
                # move accordingly
                extr[1][c] += dL  # shifted already
                extr[0][c] -= dL / 2.

        # move lower corner to (0, 0, 0)
        shift = tuple(-1. * np.array(extr[0]))
        self.translate(shift)
        self.set_cell(tuple(extr[1]))

        return shift

    def read(self, filename, format=None):
        """Read the structure from some file. The type can be given
        or it will be guessed from the filename."""

        self.__init__(read(filename, format=format))
        return len(self)
