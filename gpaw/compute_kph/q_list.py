from gpaw import GPAW
import numpy as np

# 1. Setup and Load
calc = GPAW('scf.gpw')
kpts = calc.get_bz_k_points()  # Shape: (Nk, 3)
Nk = len(kpts)
print(f'{Nk = }')

# 2. Calculate q differences
# q = k_i - k_j
q_diff = kpts[:, None, :] - kpts[None, :, :]

# 3. Enforce Periodicity (The Critical Fix)
# We map everything to the range [0, 1) using modulo arithmetic.
# This automatically unifies -0.5 and 0.5 into the same vector.
# We round to 12 decimals to ignore floating point noise.
q_periodic = np.mod(q_diff, 1.0)
q_periodic = np.round(q_periodic, decimals=12)

# Flatten to list of vectors for uniqueness check
q_flat = q_periodic.reshape(-1, 3)

# 4. Identify Unique Vectors and Indices
# return_inverse=True automatically generates the mapping matrix you were building manually
q_unique_01, indices_flat = np.unique(q_flat, axis=0, return_inverse=True)

# Reshape the flat indices back to the (Nk, Nk) matrix
q_indices = indices_flat.reshape(Nk, Nk)

# 5. (Optional) Shift q-points back to visual range [-0.5, 0.5)
# While [0, 1) is good for math, physicists usually prefer q-points centered at Gamma.
q_final = np.where(q_unique_01 > 0.5 + 1e-12, q_unique_01 - 1.0, q_unique_01)

# --- Verification ---
print(f"Total pairs: {Nk*Nk}")
print(f"Unique q-points found: {len(q_final)}") # Should be exactly 512 (for 8x8x8)
print(f"q_indices shape: {q_indices.shape}")
print(f"Max index: {q_indices.max()}")          # Should be 511

# 6. Save
np.save('q_list.npy', q_final)
np.save('q_indices.npy', q_indices)
print("Files saved: q_list.npy, q_indices.npy")
