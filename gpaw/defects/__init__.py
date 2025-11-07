import numpy as np
import numbers
from scipy.optimize import minimize
from scipy.integrate import simpson
from gpaw import GPAW, PW
from gpaw.mpi import serial_comm
from ase.parallel import parprint
import scipy.integrate as integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from ase.units import Hartree as Ha
from ase.units import Bohr
from pathlib import Path


class ElectrostaticCorrections():
    """
    Calculate the electrostatic corrections for charged defects.
    """
    def __init__(self, pristine, defect,
                 charge=None, epsilon=None, sigma=None, r0=None,
                 comm=serial_comm):

        if isinstance(pristine, (str, Path)):
            pristine = GPAW(pristine, txt=None, parallel={'domain': 1})
        if isinstance(charged, (str, Path)):
            defect = GPAW(defect, txt=None, parallel={'domain': 1})

        calc = GPAW(mode=PW(500, force_complex_dtype=True),
                    kpts={'size': (1, 1, 1),
                          'gamma': True},
                    parallel={'domain': 1},
                    symmetry='off',
                    communicator=comm,
                    txt=None)

        atoms = pristine.atoms.copy()
        calc.initialize(atoms)

        self.pristine = pristine
        self.defect = defect
        self.calc = calc
        self.charge = charge
        self.sigma = sigma
        self.epsilon = epsilon
        self.r0 = np.array(r0) / Bohr

        self.Omega = np.abs(np.linalg.det(self.calc.density.gd.cell_cv)) # volume
        # XXX check whether atoms.get_volume() gives the same?

        self.pd = self.calc.wfs.pd
        self.G_Gv = self.pd.get_reciprocal_vectors(q=0, add_q=False)
        self.G2_G = self.pd.G2_qG[0]  # |\vec{G}|^2 in Bohr^-2
        self.rho_G = self.calculate_gaussian_density()

    def calculate_gaussian_density(self):
        # Fourier transformed gaussian:
        return self.charge * np.exp(-0.5 * self.G2_G * self.sigma ** 2)

    def calculate_periodic_correction(self):
        Elp = 0.0
        for gg, G2 in enumerate(self.G_2G):
            if np.allclose(G2, 0):
                parprint('Skipping G^2=0 contribution to Elp')
            else:
                Elp += np.sum(np.abs(self.rho_G[gg])**2 / G2).real
        Elp *= 2.0 * np.pi * Ha / self.epsilon / self.Omega
        return Elp

    def calculate_isolated_correction(self):
        Eli = 0.5 * self.charge ** 2 * Ha / np.pi ** 0.5 / self.epsilon / self.sigma
        return Eli

    def calculate_potential_alignment(self):
        # XXX check sign convention
        V_neutral = - self.pristine.get_electrostatic_potential()
        V_defect = - self.defect.get_electrostatic_potential()
        V_model = self.calculate_model_potential()
        Delta_V = V_model - V_defect + V_neutral
        return Delta_V

    def calculate_model_potential(self):

        vox3 = self.calc.density.gd.cell_cv[2, :] / len(self.z_g)

        # The grid is arranged with z increasing fastest, then y
        # then x (like a cube file)

        G_z = self.G_z[1:]
        rho_Gz = self.q * np.exp(-0.5 * G_z * G_z * self.sigma * self.sigma)

        zs = []
        Vs = []
        if self.dimensionality == '2d':
            phase = np.exp(1j * (self.G_z * self.z0))
            A_GG = (self.GG * self.epsilon_GG['out-of-plane'])
            A_GG[0, 0] = 1
            V_G = np.linalg.solve(A_GG,
                                  phase * np.array([0] + list(rho_Gz)))[1:]
        elif self.dimensionality == '3d':
            phase = np.exp(1j * (G_z * self.z0))
            V_G = phase * rho_Gz / self.eb[1] / G_z ** 2

        for z in self.z_g:
            phase_G = np.exp(1j * (G_z * z))
            V = (np.sum(phase_G * V_G).real
                 * Ha * 4.0 * np.pi / (self.Omega))
            Vs.append(V)

        V = (np.sum(V_G.real) * Ha * 4.0 * np.pi / (self.Omega))
        zs = list(self.z_g) + [vox3[2]]
        Vs.append(V)
        return np.array(zs), np.array(Vs)

    def average(self, V, z):
        assert len(V) == len(z)
        N = len(V)
        deltaN = N // 8

        if self.dimensionality == '3d':
            # as far away as possible from the defect
            middle = np.argmin(np.abs(z + self.z0)) + N // 2
            middle = middle % N
        elif self.dimensionality == '2d':
            middle = 0

        points = np.arange(middle - deltaN, middle + deltaN + 1)
        points = points % N
        restricted = V[points]
        V_mean = np.mean(restricted)
        return V_mean

    def calculate_corrected_formation_energy(self):
        E_0 = self.pristine.get_potential_energy()
        E_X = self.charged.get_potential_energy()
        Eli = self.calculate_isolated_correction()
        Elp = self.calculate_periodic_correction()
        Delta_V = self.calculate_potential_alignment()
        return E_X - E_0 - (Elp - Eli) + Delta_V * self.q

    def calculate_uncorrected_formation_energy(self):
        E_0 = self.pristine.get_potential_energy()
        E_X = self.charged.get_potential_energy()
        return E_X - E_0

    def collect_electrostatic_data(self):
        V_neutral = -np.mean(self.pristine.get_electrostatic_potential(),
                             (0, 1)),
        V_charged = -np.mean(self.charged.get_electrostatic_potential(),
                             (0, 1)),
        data = {'epsilon': self.eb[0],
                'z': self.density_z,
                'V_0': V_neutral,
                'V_X': V_charged,
                'Elc': self.Elp - self.Eli,
                'D_V_mean': self.calculate_potential_alignment(),
                'V_model': self.V_model,
                'D_V': (self.V_model
                        + V_neutral
                        - V_charged)}
        self.data = data
        return data


def find_G_z(G_Gv):
    mask = (G_Gv[:, 0] == 0) & (G_Gv[:, 1] == 0)
    G_z = G_Gv[mask][:, 2]  # qG_z vectors in Bohr^{-1}
    return G_z


def find_z(gd):
    r3_xyz = gd.get_grid_point_coordinates()
    nrz = r3_xyz.shape[3]
    return r3_xyz[2].flatten()[:nrz]
