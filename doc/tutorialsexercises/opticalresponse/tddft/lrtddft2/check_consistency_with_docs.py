# This script helps to keep the documentation up-to-date.
# It checks that the values mentioned in .rst match
# with the actual output from the scripts.
#
# When updating the scripts, please update the .rst accordingly.

import numpy as np
from gpaw import GPAW


messages = []

calc = GPAW('unocc.gpw', legacy_gpaw=True)
eig_n = calc.get_eigenvalues()

ref = 42
if not len(eig_n) == ref:
    messages.append(
        f'Number of bands is not {ref}. Check that discussion makes sense.')

ref = 8.2
if not np.isclose(eig_n[-2] - eig_n[11], ref, rtol=0, atol=0.1):
    messages.append(
        f'Update the energy difference of {ref}.')


data_ij = np.loadtxt('transitions_with_08.00eV.dat')
ref = 5.9
if not np.isclose(data_ij[0, 0], ref, rtol=0, atol=0.1):
    messages.append(
        f'Update the excitation energy of {ref}.')

if len(messages) > 0:
    messages.insert(0, 'Scripts have changed. Please update the .rst too:')
    message = '\n* '.join(messages)
    raise AssertionError(message)
