# web-page: planar_averages.png
from ase.io.jsonio import read_json
import matplotlib.pyplot as plt

data = read_json('electrostatics.json')

index = data['repeats'].index(2)
profile = data['profiles'][index]
z = profile['z']
V_m = profile['model']
dV_defprs = profile['def'] - profile['prs']
dV = V_m - dV_defprs
dphi_avg = profile['dphi']

plt.plot(z, dV, '-', label=r'$\Delta V(z)$')
plt.plot(z, V_m, '-', label='$V(z)$')
plt.plot(z, dV_defprs, '-',
         label=(r'$[V^{V_\mathrm{Ga}^{-3}}_\mathrm{el}(z) -'
                r'V^{0}_\mathrm{el}(z) ]$'))

plt.axhline(dphi_avg, ls='dashed')
plt.axhline(0.0, ls='-', color='grey')
plt.xlabel(r'$z\enspace (\mathrm{\AA})$', fontsize=18)
plt.ylabel('Planar averages (eV)', fontsize=18)
plt.legend(loc='upper right')
plt.xlim((z[0], z[-1]))
plt.savefig('planar_averages.png', bbox_inches='tight', dpi=300)
