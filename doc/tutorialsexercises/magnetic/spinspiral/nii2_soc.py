import numpy as np
from gpaw.occupations import create_occ_calc
from gpaw.spinorbit import soc_eigenstates
from gpaw.new.ase_interface import GPAW


def sphere_points(distance=None):
    '''Calculates equidistant points on the upper half sphere

    Returns list of spherical coordinates (thetas, phis) in degrees

    Modified from:
        M. Deserno 2004 If Polymerforshung (Ed.) 2 99
    '''

    import math
    N = math.ceil(129600 / (math.pi) * 1 / distance**2)
    if N <= 1:
        return np.array([0.]), np.array([0.])

    A = 4 * math.pi
    a = A / N
    d = math.sqrt(a)

    # Even number of theta angles ensure 90 deg is included
    Mtheta = round(math.pi / (2 * d)) * 2
    dtheta = math.pi / Mtheta
    dphi = a / dtheta
    points = []

    # Limit theta loop to upper half-sphere
    for m in range(Mtheta // 2 + 1):
        # m = 0 ensure 0 deg is included, Mphi = 1 is used in this case
        theta = math.pi * m / Mtheta
        Mphi = max(round(2 * math.pi * math.sin(theta) / dphi), 1)
        for n in range(Mphi):
            phi = 2 * math.pi * n / Mphi
            points.append([theta, phi])
    thetas, phis = np.array(points).T

    if not any(thetas - np.pi / 2 < 1e-14):
        import warnings
        warnings.warn('xy-plane not included in sampling')

    return thetas * 180 / math.pi, phis * 180 / math.pi


energies = np.load('data.npz')['energies']
minimum = np.argmin(energies)

theta_tp, phi_tp = sphere_points(distance=1)
calc = GPAW(f'gsq-{minimum:02}.gpw')
occcalc = create_occ_calc({'name': 'fermi-dirac', 'width': 0.001})
soc_tp = np.array([])
for theta, phi in zip(theta_tp, phi_tp):
    en_soc = soc_eigenstates(calc=calc, projected=True, theta=theta, phi=phi,
                             occcalc=occcalc).calculate_band_energy()
    soc_tp = np.append(soc_tp, en_soc)

np.savez('soc_data.npz', soc=soc_tp, theta=theta_tp, phi=phi_tp)
