""" Defects module """

import numpy as np
from gpaw import GPAW, PW
from gpaw.mpi import serial_comm
from ase.parallel import parprint
from ase.units import Bohr, Hartree
from ase.geometry import find_mic
from pathlib import Path


class ElectrostaticCorrections():
    """
    Calculate the electrostatic corrections for charged defects.
    """
    def __init__(self, pristine, defect,
                 charge=None, epsilon=None, sigma=None, r0=None,
                 ravg=1.5, comm=serial_comm):

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
        self.atoms_prs = pristine.get_atoms()
        self.defect = defect
        self.calc = calc
        self.charge = charge
        self.sigma = sigma          # Bohr ? XXX consistency?
        self.epsilon = epsilon
        self.r0 = np.array(r0)      # Angstrom
        self.ravg = ravg            # Angstrom
        self.nfreq = 4              # grid coarsening
        # np.min(self.atoms_prs.cell.lengths())/8.  # Angstrom

        # volume
        self.Omega = self.atoms_prs.get_volume() / Bohr ** 3

        self.pd = self.calc.wfs.pd
        self.G_Gv = self.pd.get_reciprocal_vectors(q=0, add_q=False)
        self.G2_G = self.pd.G2_qG[0]  # |\vec{G}|^2 in Bohr^-2

    def calculate_gaussian_density(self):
        # fourier transformed gaussian:
        return self.charge * np.exp(-0.5 * self.G2_G * self.sigma ** 2)

    def calculate_gaussian_potential(self):
        phi_G = np.zeros_like(self.rho_G)
        zero = np.abs(self.G2_G) < 1e-4
        phi_G[~zero] = self.rho_G[~zero] / self.G2_G[~zero]
        return 4. * np.pi * phi_G * Hartree / self.epsilon

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
        Eli = 0.5 * self.charge ** 2 * Hartree / np.pi ** 0.5 / eps / sgm
        return Eli

    def calculate_model_potential(self, r_vR):
        # need to backtransform phi_G -> phi_r = sum_G exp(i G * r) phi_G

        # assuming G: (ng, 3), r: (3, nx, ny, nz), phi_G: (ng,)
        # compute G * r for all G and grid points
        # shape: (ng, nx, ny, nz)
        Gr = np.einsum('gi,i...->g...', self.G_Gv, r_vR)

        # compute exp(i * G * r)
        # shape: (ng, nx, ny, nz)
        exp_Gr = np.exp(1j * Gr)

        # weighted sum over G
        # shape: (nx, ny, nz)
        phi_r = np.einsum('g,g...->...', self.phi_G, exp_Gr)

        assert np.abs(phi_r.imag).max() < 1e-8

        return phi_r.real / self.Omega  # XXX right normalization ?

    def extract_electrostatic_potentials(self):
        self.phi_prs = - self.pristine.get_electrostatic_potential()
        self.phi_def = - self.defect.get_electrostatic_potential()
        finegrid = self.pristine.wfs.pd.gd.refine()
        self.r_vR = finegrid.get_grid_point_coordinates()
        finegrid_def = self.defect.wfs.pd.gd.refine()
        r_vR = finegrid_def.get_grid_point_coordinates()
        assert np.allclose(self.r_vR, r_vR)
        assert np.allclose(self.phi_prs.shape, self.phi_def.shape)
        assert np.allclose(self.phi_prs.shape, self.r_vR.shape[1:])

    def get_reference_index(self, index):
        """Get index of atom furthest away from the atom index."""

        atoms = self.atoms_prs
        dR = atoms.positions - atoms.positions[index, :][None, :]
        dist_vec, dist = find_mic(dR, atoms.get_cell())
        ref_index = np.argmax(dist)

        return ref_index

    def define_averaging_region(self):
        # locate atom farest away from the defect
        cell_prs = self.atoms_prs.get_cell()

        # in pristine obtain atom positions closest to the defect_site
        dR = self.atoms_prs.positions - self.r0[None, :]
        _, dist = find_mic(dR, cell_prs)
        defect_index = np.argmin(dist)

        # now find atom most away
        bulk_index = self.get_reference_index(defect_index)

        # return grid indices of region around the bulk atoms
        grid_shape = self.r_vR.shape[1:]
        # convert grid to Angstrom such we can use find_mic
        rgrid_vR = self.r_vR * Bohr
        rbulk_v = self.atoms_prs.positions[bulk_index, :]
        dR = rgrid_vR.T - rbulk_v[None, None, None, :]
        # flatten grid and reshape
        dR = dR.reshape((np.prod(grid_shape), 3))
        _, dist = find_mic(dR, cell_prs)
        dist = dist.reshape(grid_shape)
        # sphere radius: self.ravg
        self.region = np.where(dist < self.ravg)

    def coarsen_grid(self, nfreq):
        self.phi_prs = self.phi_prs[::nfreq, ::nfreq, ::nfreq]
        self.phi_def = self.phi_def[::nfreq, ::nfreq, ::nfreq]
        self.r_vR = self.r_vR[:, ::nfreq, ::nfreq, ::nfreq]

    def calculate_potential_profile(self):
        self.extract_electrostatic_potentials()
        self.coarsen_grid(nfreq=2)

        # restrict to z-axis
        phiz_prs = np.mean(self.phi_prs, axis=(0, 1))
        phiz_def = np.mean(self.phi_def, axis=(0, 1))
        z_vR = self.r_vR[:, :, :, :]

        # get model potential along zaxis
        phiz_model = np.mean(self.calculate_model_potential(z_vR), axis=(0, 1))

        zaxis = z_vR[2, 0, 0, :]
        iz = np.argsort(zaxis)

        profile = {'z': zaxis[iz], 'model': phiz_model[iz],
                   'prs': phiz_prs[iz], 'def': phiz_def[iz]}

        return profile

    def calculate_potential_alignment(self):
        # extract electro-static potential and grid from
        # pristine and defect
        self.extract_electrostatic_potentials()
        self.coarsen_grid(nfreq=self.nfreq)

        # define region away from defect
        self.define_averaging_region()

        # restrict to averaging region
        ix, iy, iz = self.region
        self.phi_prs = self.phi_prs[ix, iy, iz]
        self.phi_def = self.phi_def[ix, iy, iz]
        self.r_vR = self.r_vR[:, ix, iy, iz]

        # get model potential inside the averaging region
        phi_model = self.calculate_model_potential(self.r_vR)

        Delta_V = np.average(phi_model - (self.phi_def - self.phi_prs))
        return Delta_V

    def calculate_corrected_formation_energy(self):
        E_0 = self.pristine.get_potential_energy()
        E_X = self.defect.get_potential_energy()
        Eli = self.calculate_isolated_correction()
        Elp = self.calculate_periodic_correction()
        Delta_V = self.calculate_potential_alignment()
        parprint('Eli=', Eli, 'Elp=', Elp, 'Delta_V=', Delta_V)
        return E_X - E_0 - (Elp - Eli) + Delta_V * self.charge

    def calculate_uncorrected_formation_energy(self):
        E_0 = self.pristine.get_potential_energy()
        E_X = self.defect.get_potential_energy()
        return E_X - E_0
