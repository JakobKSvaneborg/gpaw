from pathlib import Path
from gpaw.mpi import world
from gpaw.response.df import DielectricFunction
from gpaw.response.qeh import QEHChiCalc

from qeh.bb_calculator.chicalc import ChiHandler
from qeh.bb_calculator.bb_builder import interpolate_chi_to_bb

df = DielectricFunction(calc='MoS2_gs_fulldiag.gpw',
                        eta=0.001,
                        frequencies={'type': 'nonlinear',
                                     'domega0': 0.05,
                                     'omega2': 10.0},
                        nblocks=8,
                        ecut=150,
                        truncation='2D')

chicalc = QEHChiCalc(df)
chihandler = ChiHandler('MoS2', chicalc, q_max=3.0)
chihandler.calculate_chi_2d()

if world.rank == 0:
    interpolate_chi_to_bb('MoS2', aN=2)
    Path('MoS2_gs_fulldiag.gpw').unlink()
