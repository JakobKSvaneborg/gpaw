from gpaw.new.ase_interface import GPAW
from ase import Atoms
from gpaw.jellium import JelliumSlab


def test_ffl():
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 8.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 3.0
    k = 2
    atoms.calc = GPAW(
        mode='pw',
        kpts=(k, k, 1),
        poissonsolver={'dipolelayer': 'xy'},
        background_charge=JelliumSlab(0.01, 4.0, 6.0))
    atoms.get_potential_energy()
    v = atoms.calc.get_electrostatic_potential()
    import matplotlib.pyplot as plt
    print(v.shape)
    plt.plot(v[0, 0])
    plt.show()


if __name__ == '__main__':
    test_ffl()
