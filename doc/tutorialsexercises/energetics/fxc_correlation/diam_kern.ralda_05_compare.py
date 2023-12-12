import numpy as np

results = np.loadtxt('diam_kern.ralda_kernel_comparison.dat')

ralda_dens = results[0] / 8
ralda_wave = results[1] / 8
RPA = results[2] / 8

assert abs(ralda_dens - -1.16) < 0.01
assert abs(ralda_wave - -1.13) < 0.01
assert abs(RPA - -1.40) < 0.01
