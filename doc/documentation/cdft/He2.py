from ase import Atoms
from gpaw import GPAW, FermiDirac, Davidson, Mixer
from gpaw.cdft.cdft import CDFT
from gpaw.cdft.cdft_coupling import CouplingParameters

# Set up the system
distance = 2.5
sys = Atoms('He2', positions=([0., 0., 0.], [0., 0., distance]))
sys.center(3)
sys.set_pbc(False)
sys.set_initial_magnetic_moments([0.5, 0.5])

# Calculator for the initial state
calc_a = GPAW(
    h=0.2,
    mode='fd',
    basis='dzp',
    charge=1,
    xc='PBE',
    symmetry='off',
    occupations=FermiDirac(0., fixmagmom=True),
    eigensolver=Davidson(3),
    spinpol=True,  # only spin-polarized calculations are supported
    nbands=4,
    mixer=Mixer(beta=0.25, nmaxold=3, weight=100.0),
    txt=f'He2+_initial_{distance:3.2f}.txt',
    convergence={
        'eigenstates': 1.0e-4,
        'density': 1.0e-1,
        'energy': 1e-1,
        'bands': 4},
    legacy_gpaw=True)

# Set initial state cdft
cdft_a = CDFT(
    calc=calc_a,
    atoms=sys,
    charge_regions=[[0]],  # choose atom 0 as the constrained region
    charges=[1],  # constrain +1 charge
    charge_coefs=[2.7],  # initial guess for Vc
    method='L-BFGS-B',  # Vc optimization method
    txt=f'He2+_initial_{distance:3.2f}.cdft',  # cDFT output file
    minimizer_options={'gtol': 0.01})  # tolerance for cdft

# Get cdft energy
sys.calc = cdft_a
sys.get_potential_energy()

# the same for the final state
calc_b = GPAW(h=0.2,
              mode='fd',
              basis='dzp',
              charge=1,
              xc='PBE',
              symmetry='off',
              occupations=FermiDirac(0., fixmagmom=True),
              eigensolver=Davidson(3),
              spinpol=True,
              nbands=4,
              mixer=Mixer(beta=0.25, nmaxold=3, weight=100.0),
              txt=f'He2+_final_{distance:3.2f}.txt',
              convergence={
                  'eigenstates': 1.0e-4,
                  'density': 1.0e-1,
                  'energy': 1e-1,
                  'bands': 4},
              legacy_gpaw=True)

cdft_b = CDFT(
    calc=calc_b,
    atoms=sys,
    charge_regions=[[1]],  # choose atom 1
    charges=[1],  # constrained charge +1
    charge_coefs=[2.7],
    method='L-BFGS-B',
    txt=f'He2+_final_{distance:3.2f}.cdft',
    minimizer_options={'gtol': 0.01})

sys.calc = cdft_b
sys.get_potential_energy()

# Now for the coupling parameter
coupling = CouplingParameters(cdft_a, cdft_b, AE=False)  # use pseudo orbitals
H12 = coupling.get_coupling_term()  # use original cDFT method
