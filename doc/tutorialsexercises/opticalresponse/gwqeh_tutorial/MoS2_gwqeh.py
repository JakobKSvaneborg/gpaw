from gpaw.response.gwqeh import GWQEHCorrection

# Define heterostructure: bilayer MoS2
structure = ['MoS2-chi.npz', 'MoS2-chi.npz']
d = [6.15]  # Interlayer distance in Angstrom

# Set up GWQEH calculation
gwq = GWQEHCorrection(calc='MoS2_fulldiag.gpw',
                      filename='MoS2_gwqeh_out',
                      kpts=[0],           # Calculate at Gamma point
                      bands=(8, 12),      # VB-2 to CB+2
                      structure=structure,
                      d=d,
                      layer=0,            # Calculate for first layer
                      domega0=0.025,      # Frequency grid parameters
                      omega2=10.0)

# Calculate the QEH self-energy contribution
gwq.calculate_QEH()

# Get quasiparticle corrections
qp_correction = gwq.calculate_qp_correction()

print('Quasiparticle corrections (eV):')
for i, band in enumerate(range(8, 12)):
    print(f'  Band {band}: {qp_correction[0, 0, i]:.4f} eV')
