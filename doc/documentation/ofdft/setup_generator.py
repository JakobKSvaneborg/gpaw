from gpaw.atom.generator import Generator

# List of elements for which setups will be generated
elements = ['N']

# Fraction of Weizsacker
lambda_coeff = 1.0

# Fraction of Thomas-Fermi
gamma_coeff = 1.0

# Select optimum cutoff and grid
for symbol in elements:
    gpernode = 800
    if symbol == 'H':
        rcut = 0.9
    elif symbol in ['He' or 'Li']:
        rcut = 1.0
    elif symbol in ['Be', 'B', 'C', 'N', 'O', 'F', 'Ne']:
        rcut = 1.2
    elif symbol in ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']:
        rcut = 1.4
    else:
        rcut = 1.0

    # If the lambda scaling is used change name to differentiate the setup
    name = f'lambda_{lambda_coeff}'

    # Use of Kinetic functional (minus the Tw contribution) inside the
    # xc definition
    pauliname = f'{gamma_coeff}_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'

    # Calculate OFDFT density
    g = Generator(symbol, xcname=pauliname, scalarrel=True,
                  orbital_free=True, tw_coeff=lambda_coeff,
                  gpernode=gpernode)

    g.run(exx=False,
          name=name,
          rcut=rcut,
          write_xml=True)
