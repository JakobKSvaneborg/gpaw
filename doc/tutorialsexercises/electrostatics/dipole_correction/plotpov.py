# creates: slab.png
from ase.build import fcc100, add_adsorbate
from ase.io import write

slab = fcc100('Al', (2, 2, 2), a=4.05, vacuum=7.5)
add_adsorbate(slab, 'Na', 4.0)
slab.center(axis=2)

write('slab.pov',
      slab,
      rotation='-90x',
      show_unit_cell=2,
      povray_settings=dict(
          transparent=False,
          display=False)).render()
