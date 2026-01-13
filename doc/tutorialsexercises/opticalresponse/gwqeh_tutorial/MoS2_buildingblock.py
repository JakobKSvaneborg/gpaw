from gpaw.response.df import DielectricFunction
from gpaw.response.qeh import QEHChiCalc

# Set up dielectric function calculator
df = DielectricFunction(calc='MoS2_fulldiag.gpw',
                        frequencies={'type': 'nonlinear',
                                     'omegamax': 15,
                                     'domega0': 0.05,
                                     'omega2': 5.0},
                        ecut=50,
                        rate=0.01,
                        truncation='2D')

# Create QEH chi calculator and save building block
chicalc = QEHChiCalc(df)
chicalc.save_chi_npz(q_max=1.5, filename='MoS2-chi.npz')

print('Building block saved to MoS2-chi.npz')
