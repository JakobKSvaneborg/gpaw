from gpaw.response.g0w0 import G0W0

gw = G0W0('C_pw_groundstate.gpw',
          integrate_gamma='WS',
          ecut=200,
          kpts=[0],
          eta=0.1, 
          bands=(3,5),
          evaluate_sigma=np.linspace(-50,75, 500),
          filename='C-g0w0-pw'
          )

gw.calculate()
