# web-page: planar_averages.png
from ase.io.jsonio import read_json
from gpaw.defects.electrostatics import plot_potentials

data = read_json('electrostatics.json')

# obtain potential profile (only generated for N=2)
profile = data['profile']
plot_potentials(profile, png='planar_averages.png')
