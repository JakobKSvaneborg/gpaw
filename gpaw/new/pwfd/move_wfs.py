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
        atomdist=atomdist)

    if desc.dtype == complex:
        disp_ac = (newfracpos_ac - oldfracpos_ac).round()
        print(disp_ac)
        phase_a = np.exp(2j * np.pi * disp_ac @ desc.kpt_c)

    P_anb = {}
    for a, B1, B2 in phit_abX.layout.myindices:
        P_anb[a] = -P_ani[a][:, :B2 - B1]

    phit_abX.add_to(psit_nX, P_anb)

    for a, P_nb in P_anb.items():
        if desc.dtype == complex:
            P_nb *= -phase_a[a]
        else:
            P_nb *= -1.0

    print(P_anb)
    phit_abX.move(newfracpos_ac, atomdist)
    phit_abX.add_to(psit_nX, P_anb)
