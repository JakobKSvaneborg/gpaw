# creates: homo-0.png, homo-1.png, homo-2.png, homo-3.png
# creates: lumo+0.png, lumo+1.png
from ase.io import write
from gpaw import GPAW
from ase.build import molecule


if 1:
    # DFT calculation:
    h2o = molecule('H2O')
    h2o.center(vacuum=3.5)
    h2o.calc = GPAW(mode={'name': 'pw', 'ecut': 800},
                    txt='h2o.txt')
    h2o.get_potential_energy()
    h2o.calc.write('h2o.gpw', mode='all')

calc = GPAW('h2o.gpw')

povray_settings = {
    'pause': False,  # Pause when done rendering (only if display)
    'transparent': True,  # Transparent background
    'canvas_width': None,  # Width of canvas in pixels
    'canvas_height': 1024,  # Height of canvas in pixels
    # 'camera_dist': 25.0,  # Distance from camera to front atom
    # 'camera_type': 'orthographic angle 35',  # 'perspective angle 20'
    'textures': len(calc.atoms) * ['ase3'],
    'celllinewidth': 0.01}  # Radius of the cylinders representing the cell
generic_projection_settings = {
    'rotation': '0x, -90y, 90z',
    'radii': len(calc.atoms) * [0.15],
    'show_unit_cell': 2}
isosurface_cutoff = 0.1

nbands = calc.get_number_of_bands()
occs = calc.get_occupation_numbers()
homo = (occs > 1.0).sum() - 1


for band in range(nbands):
    if band <= homo:
        name = f'homo-{homo - band}'
    else:
        name = f'lumo+{band - homo - 1}'
    if calc.old:
        wf = calc.get_pseudo_wave_function(band=band)
    else:
        wf = calc.dft.wave_function(band).data
    isosurface_data = []
    if wf.max() >= isosurface_cutoff:
        isosurface_data.append({
            'density_grid': wf,
            'cut_off': isosurface_cutoff,
            'closed_edges': False,
            'color': [0.25, 0.25, 0.80, 0.5],
            'material': 'simple'})
    if wf.min() <= -isosurface_cutoff:
        isosurface_data.append({
            'density_grid': wf,
            'cut_off': -isosurface_cutoff,
            'closed_edges': False,
            'color': [0.80, 0.25, 0.25, 0.5],
            'material': 'simple'})
    write(f'{name}.pov',
          calc.atoms,
          **generic_projection_settings,
          povray_settings=povray_settings,
          isosurface_data=isosurface_data).render()
