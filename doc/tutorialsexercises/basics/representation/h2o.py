from ase.build import molecule
from gpaw import GPAW

a = 8.0
ecut = 400

energies = {}
with open(f'results-{ecut:3.0f}.txt', 'w') as resultfile:

    for name in ['H2O', 'H', 'O']:
        atoms = molecule(name)
        atoms.set_cell((a, a, a))
        atoms.center()

        if name in ['H', 'O']:
            hund = True
        else:
            hund = False

        calc = GPAW(mode={'name': 'pw', 'ecut': ecut}, hund=hund,
                    txt=f'gpaw-{name}-{ecut:3.0f}.txt')
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        energies[name] = energy
        print(name, energy, file=resultfile)

    e_atomization = energies['H2O'] - 2 * energies['H'] - energies['O']
    print(e_atomization, file=resultfile)
