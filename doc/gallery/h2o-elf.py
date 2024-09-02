# creates: h2o-elf.png
import plotly.graph_objects as go
from ase.build import molecule
from gpaw.new.ase_interface import GPAW
from gpaw.elf import elf_from_dft_calculation

h2o = molecule('H2O')
h2o.center(vacuum=3.5)
h2o.calc = GPAW(mode={'name': 'pw', 'ecut': 1000},
                txt='h2o.txt')
h2o.get_potential_energy()
elf_R = elf_from_dft_calculation(h2o.calc.dft, ncut=0.001)

# Plot:
surf = elf_R.isosurface(show=False, isomin=0.8, isomax=0.8)
fig = go.Figure(data=[surf])
fig.update_traces(showscale=False)
fig.update_layout(
    scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
               camera=dict(up=dict(x=0.8, y=0.4, z=0.8),
                           eye=dict(x=0.5, y=0.3, z=0.1))))
fig.write_image('h2o-elf.png')
