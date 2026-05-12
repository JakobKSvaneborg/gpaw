# web-page: W_r.png
import numpy as np
import matplotlib.pyplot as plt
from qeh import QEH


thick_MoS2 = 6.2926
thick_WSe2 = 6.718

d_MoS2_WSe2 = (thick_MoS2 + thick_WSe2) / 2
inter_mass = 0.244

HS = QEH.heterostructure(BBfiles=['MoS2-bb-int', 'WSe2-bb-int'],
                         layerwidth_n=[thick_MoS2, thick_WSe2],
                         wmax=0,
                         amax=2)

hl_array = np.array([0., 0., 1., 0.])
el_array = np.array([1., 0., 0., 0.])


# Getting the interlayer exciton screened interaction on a real grid
r, W_r = HS.get_exciton_screened_potential_r(
    r_array=np.linspace(1e-1, 30, 1000),
    e_distr=el_array,
    h_distr=hl_array,
    intralayer=False,)

plt.plot(r, W_r, '-g')
plt.title(r'Screened Interaction Energy')
plt.xlabel(r'${\bf r}$ (Ang)', fontsize=20)
plt.ylabel(r'$W({\bf r})$ (Ha)', fontsize=20)
plt.savefig('W_r.png')
plt.show()

ee, ev = HS.get_exciton_binding_energies(eff_mass=inter_mass,
                                         e_distr=el_array,
                                         h_distr=hl_array,
                                         intralayer=False,)

print('The interlayer exciton binding energy is:', -ee[0])
