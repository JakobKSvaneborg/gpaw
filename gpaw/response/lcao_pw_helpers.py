import numpy as np
from ase.units import Ha, Bohr 

def pw_ecut_from_lcao_grid(gd) -> float:
    """Ecut,PW=Gmax^2/2. Link to real-space grid with hmax: Gmax=π/h_max -> Ecut≈1/4*(π/h_max)^2 """
    if hasattr(gd,"h_c"):
        h_max = np.max(gd.h_c)
    else:  # non-orthorhombic 
        h_max = np.max(np.linalg.norm(np.asarray(gd.h_cv), axis=1))
    ecut_pw = Ha*1/4*(np.pi/h_max)**2
    return ecut_pw
