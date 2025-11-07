""" Defects module """

import numpy as np
from gpaw import GPAW, PW
from gpaw.mpi import serial_comm
from ase.parallel import parprint
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
        if isinstance(defect, (str, Path)):
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

        # volume
        self.Omega = np.abs(np.linalg.det(self.calc.density.gd.cell_cv))
        # XXX check whether atoms.get_volume() gives the same?

        self.pd = self.calc.wfs.pd
        self.G_Gv = self.pd.get_reciprocal_vectors(q=0, add_q=False)
        self.G2_G = self.pd.G2_qG[0]  # |\vec{G}|^2 in Bohr^-2

    def calculate_gaussian_density(self):
        # fourier transformed gaussian:
        return self.charge * np.exp(-0.5 * self.G2_G * self.sigma ** 2)

    def calculate_gaussian_potential(self):
        phi_G = np.zeros_like(self.rho_G)
        for gg, G2 in enumerate(self.G_2G):
            if np.allclose(G2, 0):
                parprint('Skipping G^2=0 contribution to Elp')
                # neutralizing background cancels contribution
                phi_G[gg] = 0.0
            else:
                phi_G[gg] = self.rho_G[gg] / G2
        return 4. * np.pi * phi_G * Ha / self.epsilon

    def calculate_periodic_correction(self):
        self.rho_G = self.calculate_gaussian_density()
        self.phi_G = self.calculate_gaussian_potential()
        # electro-static energy of model charge distribution
        # interacting with its images
        # (and itself -> needs to be substracted
        # with calculate_isolated_correction)
        # neutralizing background taken into account
        Elp = 0.5 * np.sum(self.rho_G * self.phi_G).real / self.Omega
        return Elp

    def calculate_isolated_correction(self):
        # electro-static self-interaction energy of the
        # gaussian model charge distribution
        eps = self.epsilon
        sgm = self.sigma
        Eli = 0.5 * self.charge ** 2 * Ha / np.pi ** 0.5 / eps / sgm
        return Eli

    def calculate_potential_alignment(self):
        # XXX check sign convention
        V_neutral = - self.pristine.get_electrostatic_potential()
        V_defect = - self.defect.get_electrostatic_potential()
        # V_model = self.calculate_model_potential()
        V_model = 0
        Delta_V = V_model - V_defect + V_neutral
        return Delta_V

    def calculate_model_potential(self):
        # need to backtransform phi_G -> phi_r
        rvec, phi_r = None, None
        return rvec, phi_r

    def calculate_corrected_formation_energy(self):
        E_0 = self.pristine.get_potential_energy()
        E_X = self.charged.get_potential_energy()
        Eli = self.calculate_isolated_correction()
        Elp = self.calculate_periodic_correction()
        Delta_V = self.calculate_potential_alignment()
        print('Eli=', Eli, 'Elp=', Elp, 'Delta_V=', Delta_V)
        return E_X - E_0 - (Elp - Eli) + Delta_V * self.charge

    def calculate_uncorrected_formation_energy(self):
        E_0 = self.pristine.get_potential_energy()
        E_X = self.charged.get_potential_energy()
        return E_X - E_0
