from gpaw.xc.rpa import RPACorrelation

# This calculation is too heavy to run as an exercise!!

rpa1 = RPACorrelation('si_rpa_isolated.gpw', txt='si_atom_rpa_output.txt',
                      ecut=400.0, nblocks=16)
E1_i = rpa1.calculate()
