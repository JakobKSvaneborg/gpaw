from gpaw.response.g0w0 import G0W0

gw = G0W0('C_lcao_groundstate.gpw',
          nbands=((1 + 3) * 2 + 5) * 2,
          integrate_gamma='WS',
          ecut=200,
          kpts=[0],
          eta=0.1, bands=(0, 8))

gw.calculate()
