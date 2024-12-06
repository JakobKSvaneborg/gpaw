from gpaw.utilities.folder import Folder

import numpy as np

import pytest

import matplotlib.pyplot as plt


def test_vering_fold():
    x = [1, 3, 5, 7, 9]
    y = [2, 6, 1, 9, 3]

    width = 0.2
    linbroad = [0.4, 4, 8]
    for folding in ['Gauss', 'Lorentz']:
        x_c, y_c = Folder(width, folding).fold(x, y)

        x_v, y_v = Folder(width, folding).fold(x, y, linbroad=linbroad)
        assert (x_c == x_v).all()

        i = np.where(x_c < 4)
        assert y_c[i] == pytest.approx(y_v[i], abs=1e-01)

        i2 = np.where(x_c > 8)
        assert max(y_c[i2]) > max(y_v[i2])

        # XXX Remove later
        plt.plot(x_c, y_c)
        plt.plot(x_v, y_v)

        # plt.show()
