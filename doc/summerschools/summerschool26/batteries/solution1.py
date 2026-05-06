# creates: layerdist.txt
# This script will calculate the energy of graphite for a series of
# inter-layer distances.
import json
from pathlib import Path

import numpy as np
from ase.calculators.emt import EMT
from ase.lattice.hexagonal import Graphite

# ccdist is already defined in the previous cell
ccdist = ...
# Start from a small guess
layerdist = 2.0

# Teacher
dists = np.linspace(layerdist * .8, layerdist * 1.2, 10)
# The resulting energies are put in the following list
energies = []
for ld in dists:
    # Calculate the lattice parameters a and c from the C-C distance
    # and interlayer distance, respectively
    a = ccdist * np.sqrt(3)
    c = 2 * ld

    gra = Graphite('C', latticeconstant={'a': a, 'c': c})

    # Set the calculator and attach it to the Atoms object with
    # the graphite structure
    calc = EMT()
    gra.calc = calc

    energy = gra.get_potential_energy()
    # Append results to the list
    energies.append(energy)

poly = np.polynomial.Polynomial.fit(dists, energies, 3)
layerdist = next(d for d in poly.deriv().roots() if poly.deriv(2)(d) > 0)
Path('layerdist.json').write_text(json.dumps({'distance': layerdist}))
