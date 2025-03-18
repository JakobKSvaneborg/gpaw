from gpaw.new.ase_interface import GPAW
from ase import Atoms
from gpaw.jellium import JelliumSlab
import numpy as np
from gpaw.mixer import Mixer
from gpaw import FermiDirac


def test_ffl():
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 11.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    k = 4
    atoms.calc = GPAW(
        mode='pw',
        kpts=(k, k, 1),
        occupations=FermiDirac(0.2),
        convergence={'minimum iterations': 120},  # 'density': 1.0},
        poissonsolver={'dipolelayer': 'xy'},
        background_charge=dict(charge=0.0001,
                               z1=7.0, z2=9.0,
                               fermi_level=-3.15),
        )#txt=None)
    atoms.get_potential_energy()
    v = atoms.calc.get_electrostatic_potential()
    import matplotlib.pyplot as plt
    plt.plot(np.linspace(0, 11, v.shape[2], 0), v[0, 0])
    plt.show()


def test_ffl2():
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 11.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    k = 4
    E = np.linspace(-0.01, 0.01, 6)
    F = []
    for e in E:
        atoms.calc = GPAW(
            mode='pw',
            kpts=(k, k, 1),
            occupations=FermiDirac(0.2),
            mixer=Mixer(0.001, 1),
            convergence={'density': 1.0},
            poissonsolver={'dipolelayer': 'xy'},
            background_charge=dict(charge=e,
                                   z1=7.0, z2=9.0,
                                   fermi_level=None),
            txt=None)
        atoms.get_potential_energy()
        if 0:
            v = atoms.calc.get_electrostatic_potential()
            import matplotlib.pyplot as plt
            plt.plot(np.linspace(0, 11, v.shape[2], 0), v[0, 0])
            plt.show()
        F.append(atoms.calc.get_fermi_level())
        print('XXX', e, atoms.calc.get_fermi_level())
        print('XXX', atoms.calc.eigenvalues())
    import matplotlib.pyplot as plt
    plt.plot(E, F)
    plt.show()


if __name__ == '__main__':
    test_ffl()
