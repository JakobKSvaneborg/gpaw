# web-page: pdos.png
from gpaw import GPAW, restart
import matplotlib.pyplot as plt
from gpaw.utilities.dos import all_electron_LDOS, fold

# Density of States
plt.subplot(211)
slab, calc = restart('top.gpw')
doscalc = calc.dos()
energies = doscalc.get_energies(emin=-15, emax=10, npoints=501)
dos = doscalc.raw_dos(energies, width=0.2)
e_f = calc.get_fermi_level()
plt.plot(energies, dos)
plt.ylabel('DOS')

molecule = range(len(slab))[-2:]

plt.subplot(212)
c_mol = GPAW('CO.gpw')
for n in range(2, 7):
    print('Band', n)
    # PDOS on the band n
    wf_k = [kpt.psit_nG[n] for kpt in c_mol.wfs.kpt_u]
    P_aui = [[kpt.P_ani[a][n] for kpt in c_mol.wfs.kpt_u]
             for a in range(len(molecule))]
    e, w = all_electron_LDOS(
        calc, molecule,
        spin=0, wf_k=wf_k, P_aui=P_aui)
    e, dos = fold(e, w, npts=2000, width=0.2)
    plt.plot(e - e_f, dos, label='Band: ' + str(n))
plt.legend()
plt.axis([-15, 10, None, None])
plt.xlabel('Energy [eV]')
plt.ylabel('All-Electron PDOS')
plt.savefig('pdos.png')
plt.show()
