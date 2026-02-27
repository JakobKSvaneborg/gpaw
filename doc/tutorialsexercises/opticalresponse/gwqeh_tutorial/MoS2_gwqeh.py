from gpaw.response.gwqeh import GWQEHCorrection

# Define heterostructure: bilayer MoS2/MoS2
# Each entry in 'structure' is a dielectric building block file.
# 'd' contains the interlayer distances (one fewer than layers).
structure = ['MoS2-chi.npz', 'MoS2-chi.npz']
d = [6.15]  # Interlayer distance in Angstrom

# Set up G Delta-W QEH calculation
gwq = GWQEHCorrection(calc='MoS2_fulldiag.gpw',
                      filename='MoS2_gwqeh_out',
                      kpts=[0],           # Gamma point (correction is ~k-independent)
                      bands=(8, 12),      # 2 valence + 2 conduction bands
                      structure=structure,
                      d=d,
                      layer=0,            # Compute correction for first layer
                      domega0=0.025,      # Frequency grid: min spacing (eV)
                      omega2=10.0)        # Frequency grid: doubling energy (eV)

# Calculate quasiparticle correction (runs self-energy calculation internally)
qp_correction = gwq.calculate_qp_correction()

print('G Delta-W QP corrections (eV):')
for i, band in enumerate(range(8, 12)):
    print(f'  Band {band}: {qp_correction[0, 0, i]:.4f} eV')
