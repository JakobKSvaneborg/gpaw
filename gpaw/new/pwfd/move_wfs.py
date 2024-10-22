import numpy as np


def move_wave_functions(oldfracpos_ac,
                        newfracpos_ac,
                        atomdist,
                        P_ani,
                        psit_nX,
                        setups):
    desc = psit_nX.desc
    phit_abX = desc.atom_centered_functions(
        [setup.get_partial_waves_for_atomic_orbitals() for setup in setups],
        oldfracpos_ac,
        atomdist=atomdist,
        xp=psit_nX.xp)

    P_anb = {}
    for a, B1, B2 in phit_abX.layout.myindices:
        P_anb[a] = -P_ani[a][:, :B2 - B1]

    phit_abX.add_to(psit_nX, P_anb)

    if desc.dtype == complex:
        disp_ac = (newfracpos_ac - oldfracpos_ac).round()
        phase_a = np.exp(2j * np.pi * disp_ac @ desc.kpt_c)
        for a, P_nb in P_anb.items():
            P_nb *= -phase_a[a]
    else:
        P_anb.data *= -1.0

    phit_abX.move(newfracpos_ac, atomdist)
    phit_abX.add_to(psit_nX, P_anb)
