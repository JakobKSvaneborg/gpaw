import numpy as np
from ase.build import mx2

from gpaw import GPAW, PW, FermiDirac
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.nlopt.shg import get_shg

# Make the structure and add the vaccum around the layer
atoms = mx2(formula='MoS2', a=3.16, thickness=3.17, vacuum=5.0)
atoms.center(vacuum=15, axis=2)

# GPAW parameters:
nk = 40
params_gs = {
    'mode': PW(800),
    'symmetry': {'point_group': False, 'time_reversal': True},
    'nbands': 'nao',
    'convergence': {'bands': -10},
    'parallel': {'domain': 1},
    'occupations': FermiDirac(width=0.05),
    'kpts': {'size': (nk, nk, 1), 'gamma': True},
    'xc': 'PBE',
    'txt': 'gs.txt'
}

# Ground state calculation:
gs_name = 'gs.gpw'
calc = GPAW(**params_gs)
atoms.calc = calc
atoms.get_potential_energy()
atoms.calc.write(gs_name, mode='all')
# GSEnd

# Calculate momentum matrix elements:
nlodata = make_nlodata(gs_name)
nlodata.write('mml.npz')
# MMECalcEnd

# Shift parameters:
eta = 0.05  # Broadening in eV
w_ls = np.linspace(0, 6, 500)  # in eV
pol = 'yyy'

# LG calculation
shg_name1 = 'shg_' + pol + '_lg.npy'
get_shg(
    nlodata, freqs=w_ls, eta=eta, pol=pol,
    gauge='lg', out_name=shg_name1)

# VG calculation
shg_name2 = 'shg_' + pol + '_vg.npy'
get_shg(
    nlodata, freqs=w_ls, eta=eta, pol=pol,
    gauge='vg', out_name=shg_name2)
