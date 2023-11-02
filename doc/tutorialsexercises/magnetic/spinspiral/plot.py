# web-page: e-spiral.png
import numpy as np
import matplotlib.pyplot as plt
from ase.spectrum.band_structure import BandStructure
from ase.build import mx2

atoms = mx2('NiI2', kind='1T', a=3.969662131560825,
            thickness=3.027146598949815, vacuum=4)

data = np.load('data.npz')
energies = data['energies']
magmoms = data['magmoms']

path = atoms.cell.bandpath('GMKG', npoints=31)

energies = (energies - energies[0]) * 1000
bs = BandStructure(path, energies[np.newaxis, :, np.newaxis])
bs.plot(emin=-30, emax=45, ylabel='Energy [meV per atom]')
plt.savefig('e-spiral.png')
