import numpy as np
from ase.units import _e
from ase.dft.kpoints import monkhorst_pack
from gpaw import GPAW
from gpaw.berryphase import parallel_transport
from gpaw.mpi import world

phi_i = []

for i in range(8):
    kpts = monkhorst_pack((8, 1, 8))
    kpts += (1 / 16, -3 / 8, 1 / 16)
    kpts += (0, i / 8, 0)

    calc = GPAW('BaTiO3.gpw').fixed_density(symmetry='off', kpts=kpts)
    calc.write('BaTiO3_%s.gpw' % i, mode='all')

    phi_km, S_km = parallel_transport('BaTiO3_%s.gpw' % i, direction=2,
                                      name='BaTiO3_%s' % i, scale=0)

    phi_km[np.where(phi_km < 0.0)] += 2 * np.pi
    phi_i.append(np.sum(phi_km) / len(phi_km))
#    if world.rank == 0:
#        import matplotlib.pyplot as plt
#        for phi_k in phi_km.T:
#            plt.plot(range(8), phi_k, 'o', c='C0')
#        plt.show()

spos_ac = calc.atoms.get_scaled_positions()
Z_a = [10, 12, 6, 6, 6]  # Charge of nucleii

phase = -np.sum(phi_i) / len(phi_i)
phase += 2 * np.pi * np.dot(Z_a, spos_ac)[2]

cell_cv = calc.atoms.cell * 1e-10
V = calc.atoms.get_volume() * 1e-30
pz = (phase / (2 * np.pi) % 1) * cell_cv[2, 2] / V * _e
if world.rank == 0:
    print(f'P: {pz} C/m^2')
    with open('parallel_BaTiO3.out', 'w') as fd:
        print(f'P: {pz} C/m^2', file=fd)
