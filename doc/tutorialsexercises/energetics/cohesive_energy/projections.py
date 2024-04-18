from gpaw import GPAW
from gpaw.spherical_harmonics import names
from ase.units import Ha

calc = GPAW('pt-atom.gpw')
setup = calc.setups[0]
labels = [f'{n}{"spd"[l]}({names[l][m]})'
          for n, l in zip(setup.n_j, setup.l_j)
          for m in range(2 * l + 1)]

lines = ['#,eig. [eV],occ,character,eig. [eV],occ,character']
for n in range(11):
    line = str(n)
    for spin in [0, 1]:
        kpt = calc.wfs.kpt_qs[0][spin]
        P_i = kpt.P_ani[0][n]
        i = abs(P_i).argmax()
        label = labels
        eig = kpt.eps_n[n] * Ha
        occ = kpt.f_n[n]
        line += f',{eig:.3f},{occ:.1f},{labels[i]}'
    lines.append(line)

# Write csv-file:
with open('pt-atom.csv', 'w') as fd:
    print('\n'.join(lines), file=fd)
