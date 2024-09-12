from gpaw.lrtddft import LrTDDFT


lr = LrTDDFT.read('lr.dat.gz')
lr.diagonalize()
print('Full matrix gives', len(lr), 'excitations')

# now diagonalize using a selectron of orbitals
restrict = {'energy_range': 5}  # eV
lr.diagonalize(restrict=restrict)
print('Restriction by', restrict, 'gives', len(lr), 'excitation')

restrict = {'to': [3, 4]}
lr.diagonalize(restrict=restrict)
print('Restriction by', restrict, 'gives', len(lr), 'excitations')
