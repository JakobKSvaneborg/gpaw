import matplotlib.pyplot as plt
from ase.units import Bohr
from gpaw.new.ase_interface import GPAW

calc = GPAW('hli.gpw', txt=None)
elpot = calc.dft.electrostatic_potential()

# Avarage PS potentials:
vh, vli = elpot.atomic_potentials()
zh, zli = calc.atoms.positions[:, 2]
rh, rli = [max(setup.rcut_j) * Bohr for setup in elpot.setups]
plt.plot([zh - rh, zh + rh], [vh, vh], label=r'$\tilde v$(H)')
plt.plot([zli - rli, zli + rli], [vli, vli], label=r'$\tilde v$(Li)')

# Interpolated PS and AE potentials:
ps = elpot.pseudo_potential(grid_spacing=0.025)
ae = elpot.all_electron_potential(grid_spacing=0.025)
i = ps.data.shape[0] // 2

x, y = ps.xy(i, i, ...)
plt.plot(x, y, '-', label=r'$\tilde v$')

x, y = ae.xy(i, i, ...)
plt.plot(x, y, '-', label=r'$v$')

plt.plot(x, 0 * x, 'k')

plt.xlabel('z [Ang]')
plt.ylabel('potential [eV]')
plt.ylim(bottom=-100, top=10)
plt.legend()
plt.savefig('hli-pot.png')
