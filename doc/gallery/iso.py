from ase.io import write
from ase.dft.bandgap import bandgap
from gpaw.core import UGArray, UGDesc


#from gpaw.new.ase_interface import GPAW
from gpaw import GPAW
# creates: -H2O
from ase.build import molecule


if 0:
    # DFT calculation:
        h2o = molecule('H2O')
        h2o.center(vacuum=3.5)
        h2o.calc = GPAW(mode={'name': 'pw', 'ecut': 1000},
                        txt='h2o.txt')
        h2o.get_potential_energy()

        h2o.calc.write('H2O.gpw', mode='all')

calc = GPAW('H2O.gpw')


isosurface_cutoff = 0.1

gridrefinement=1 #once interpolation is implementeds

#rotation = '24x, 34y, 14z' # Euler angles that match ASE GUI -> View -> Rotate
rotation = '0x, -90y, 90z'


def interpolate_wf_UGArray(wf, gridrefinement=1):
    #if gri
    new_grid = UGDesc(
        cell = wf.cell_cv,# old_calc.atoms.get_cell(),
        size = np.array(wf.size)*gridrefinement )
    #print('fine grid', np.array(elf_data.data.shape)*4)
    wf_interp = wf.interpolate(grid=new_grid)
    return wf_interp

#def interpolate_wf_NDArray():
#    ...TODO

###########

povray_settings = {
    # For povray files only
    'pause': False,  # Pause when done rendering (only if display)
    'transparent': True,  # Transparent background
    'canvas_width': None,  # Width of canvas in pixels
    'canvas_height': 1024,  # Height of canvas in pixels
    #'camera_dist': 25.0,  # Distance from camera to front atom
    #'camera_type': 'orthographic angle 35',  # 'perspective angle 20'
    'textures': len(calc.atoms) * ['ase3'],
    'celllinewidth': 0.01, # Radius of the cylinders representing the cell
    }

# some more options:
# 'image_plane'  : None,  # Distance from front atom to image plane
#                         # (focal depth for perspective)
# 'camera_type'  : 'perspective', # perspective, ultra_wide_angle
# 'point_lights' : [],             # [[loc1, color1], [loc2, color2],...]
# 'area_light'   : [(2., 3., 40.) ,# location
#                   'White',       # color
#                   .7, .7, 3, 3], # width, height, Nlamps_x, Nlamps_y
# 'background'   : 'White',        # color
# 'textures'     : tex, # Length of atoms list of texture names
# 'celllinewidth': 0.05, # Radius of the cylinders representing the cell


# for plotting variables
generic_projection_settings = {
    'rotation': rotation,
    'radii': len(calc.atoms) * [0.15],
    'show_unit_cell': 2}


#############

gap, p1, p2 = bandgap(calc)
homo = p1[2]
lumo = p2[2]

nbands = calc.get_number_of_bands()

for number in range(nbands):
    if number <= homo:
        name = 'homo_{:04d}'.format(number-homo)
    else:
        name = 'lumo_{:04d}'.format(number-lumo)
    print(name)

    wf = calc.get_pseudo_wave_function(band=number)
    print(wf.shape)

    # if we have an interpotaion scheme it could go here.
    #wf_interp = interpolate_wf(wf)
    #print(wf_interp.size)


    isosurface_data=[]

    if wf.max()>= isosurface_cutoff:
        isosurface_data.append({
            'density_grid': wf,
            'cut_off'     : isosurface_cutoff,
            'closed_edges': False,
            'color'       : [0.25, 0.25, 0.80, 0.5],
            'material'    :'simple'})
    if wf.min() <= -isosurface_cutoff:
        isosurface_data.append({
            'density_grid': wf,
            'cut_off'     : -isosurface_cutoff,
            'closed_edges': False,
            'color'       : [0.80, 0.25, 0.25, 0.5],
            'material'    :'simple'})


    write('H2O_{}.pov'.format(name),
        calc.atoms,
        **generic_projection_settings,
        povray_settings=povray_settings,
        isosurface_data=isosurface_data
        ).render()
