from gpaw import GPAW


calc = GPAW('O.gpw', txt=None)
lumo = calc.get_pseudo_wave_function(band=2, spin=1)
assert calc.get_number_of_bands() == 4
