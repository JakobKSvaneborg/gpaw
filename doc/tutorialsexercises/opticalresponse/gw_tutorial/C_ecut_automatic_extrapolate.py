from ase.build import bulk

from gpaw.response.g0w0 import G0W0

a = 3.567
atoms = bulk('C', 'diamond', a=a)

for ecut in [100, 200, 300, 400]:
    gw = G0W0(calc='C_groundstate_8.gpw',
              bands=(3, 5),
              ecut=ecut,
              ecut_extrapolation=True,
              kpts=[0],
              integrate_gamma='WS',
              filename=f'C-g0w0_k8_ecut_{ecut}_automatic_extrapolate')

    result = gw.calculate()
