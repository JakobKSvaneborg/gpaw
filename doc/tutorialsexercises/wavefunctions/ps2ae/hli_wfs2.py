import matplotlib.pyplot as plt
from ase import Atoms
from ase.units import Bohr
from gpaw.new.ase_interface import GPAW

plt.rcParams['font.size'] = 12

hli = Atoms('HLi', positions=[[0, 0, 0], [0, 0, 1.6]])
hli.center(vacuum=2.5)
hli.calc = GPAW(txt='hli.txt', mode='fd')
hli.get_potential_energy()

for n in range(2):
    ae = hli.calc.dft.state.ibzwfs.get_all_electron_wave_function(
        n, grid_spacing=0.05)
    norm_squared = ae.norm2()
    print('Norm:', norm_squared)
    assert abs(norm_squared - 1) < 1e-2

    # Interpolated AE wfs:
    i0, i1 = ae.desc.size[:2] // 2
    x, y = ae.xy(i0, i1, ...)
    x *= Bohr
    y *= Bohr**-1.5
    plt.plot(x, y, '-', color=f'C{n}', label=rf'$\psi_{n}$')

    # Raw PS wfs:
    ps = hli.calc.dft.state.ibzwfs.wfs_qs[0][0].psit_nX[n]
    i0, i1 = ps.desc.size[:2] // 2 - 1
    x, y = ps.xy(i0, i1, ...)
    x *= Bohr
    y *= Bohr**-1.5
    plt.plot(x, y, 'o', color=f'C{n}', label=rf'$\tilde\psi_{n}$')

plt.hlines(0, xmin=x[0], xmax=x[-1])
plt.xlabel(r'z [$\rm \AA$]')
plt.ylabel(r'wave functions [$\rm \AA^{-3/2}$]')
plt.legend()
plt.tight_layout()
plt.savefig('hli-wfs.png')
hli.calc.write('hli.gpw')
plt.close()
