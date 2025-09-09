import numpy as np
from ase.units import Ha, Bohr # Ha (eV), Bohr (Å)

def pw_ecut_from_lcao_grid(gd) -> float:
    """Ecut,PW=Gmax^2/2. Link to real-space grid with hmax: Gmax≈π/h_max -> Ecut=(π/h_max)^2/2 """
    if hasattr(gd,"h_c"):
        h_max = np.max(gd.h_c)
    else:  # non-orthorhombic 
        h_max = np.max(np.linalg.norm(np.asarray(gd.h_cv), axis=1))
    h_max_bohr = h_max/Bohr 
    ecut_Ha = 0.5*(np.pi/h_max_bohr)**2
   #ecut_eV = ecut_Ha*Ha
    return ecut_Ha
