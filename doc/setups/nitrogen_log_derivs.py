# creates: nitrogen-log-derivs.png
from gpaw.atom.generator2 import generate, plot_log_derivs
import matplotlib.pyplot as plt

kwargs = {'symbol': 'N',
          'Z': 7,
          'xc': 'PBE',
          'projectors': '2s,s,2p,p,d,F',
          'radii': [1.3],
          'scalar_relativistic': True,
          'r0': 1.1,
          'v0': None,
          'nderiv0': 5,
          'pseudize': ('poly', 4)}

g = generate(**kwargs)
plot_log_derivs(g, 'spdfg', True, plt.gca())
plt.savefig('nitrogen-log-derivs.png')
