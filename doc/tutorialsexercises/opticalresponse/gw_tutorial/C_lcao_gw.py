import numpy as np
from gpaw.new.calculation import DFTCalculation
from gpaw.response.g0w0 import G0W0

dft = DFTCalculation.from_gpw_file('C_lcao_groundstate.gpw')
dft.change_mode('pw')
dft.write_gpw_file('C_pw_from_lcao_groundstate.gpw', include_wfs=True)

gw = G0W0('C_pw_from_lcao_groundstate.gpw',
          integrate_gamma='WS',
          ecut=200,
          nbands=26,
          kpts=[0],
          eta=0.1,
          bands=(3, 5),
          evaluate_sigma=np.linspace(-50, 75, 500),
          filename='C-g0w0-lcao')

gw.calculate()
