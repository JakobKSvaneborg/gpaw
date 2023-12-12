import numpy as np

results = np.load('formation_energies.npz')
repeats = results['repeats']
idx_222 = np.where(repeats == 2)[0][0]
diff_222 = (results['corrected'] - results['uncorrected'])[idx_222]
assert abs(diff_222 - 1.69) < 0.01

El = np.load('electrostatic_data_111.npz')
assert abs(El['D_V_mean'] - -0.914) < 0.01
