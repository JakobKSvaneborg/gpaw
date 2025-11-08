""" Defects module """

import numpy as np
from gpaw import GPAW, PW
from gpaw.mpi import serial_comm
from ase.parallel import parprint
from ase.units import Bohr, Hartree
from ase.geometry import find_mic
from pathlib import Path
import time


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'#timing {func.__name__} {end - start:.5f} seconds')
        return result
    return wrapper


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
        self.cell_prs = self.atoms_prs.get_cell()
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

        # potential alignment
        self.phi_prs = None
        self.phi_def = None
        self.dphi = None

    def calculate_gaussian_density(self):
        # fourier transformed gaussian:
        return self.charge * np.exp(-0.5 * self.G2_G * self.sigma ** 2)

    def calculate_gaussian_potential(self):
        phi_G = np.zeros_like(self.rho_G)
        zero = np.abs(self.G2_G) < 1e-4
        phi_G[~zero] = self.rho_G[~zero] / self.G2_G[~zero]
        return 4. * np.pi * phi_G * Hartree / self.epsilon

    @timeit
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

    @timeit
    def calculate_isolated_correction(self):
        # electro-static self-interaction energy of the
        # gaussian model charge distribution
        eps = self.epsilon
        sgm = self.sigma
        Eli = 0.5 * self.charge ** 2 * Hartree / np.pi ** 0.5 / eps / sgm
        return Eli

    @timeit
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

    @timeit
    def extract_electrostatic_potentials(self):
        if self.phi_prs is not None:
            return
        self.phi_prs = - self.pristine.get_electrostatic_potential()
        self.phi_def = - self.defect.get_electrostatic_potential()
        finegrid = self.pristine.wfs.pd.gd.refine()
        self.r_vR = finegrid.get_grid_point_coordinates()
        self.ng_v = np.array(self.r_vR.shape[1:])
        finegrid_def = self.defect.wfs.pd.gd.refine()
        r_vR = finegrid_def.get_grid_point_coordinates()
        assert np.allclose(self.r_vR, r_vR)
        assert np.allclose(self.phi_prs.shape, self.phi_def.shape)
        assert np.allclose(self.phi_prs.shape, self.ng_v)

    def prs_mic_dist(self, r_v):

        atoms = self.atoms_prs
        dR = atoms.positions - r_v[None, :]
        _, dist = find_mic(dR, self.cell_prs)

        return dist

    def grid_mic_dist(self, r_v):
        grid_shape = self.ngc_v
        # convert grid to Angstrom such we can use find_mic
        rg_vR = self.rc_vR * Bohr

        dR = rg_vR.T - r_v[None, None, None, :]
        # flatten grid and reshape
        dR = dR.reshape((np.prod(grid_shape), 3))
        _, dist = find_mic(dR, self.cell_prs)
        dist = dist.reshape(grid_shape)

        return dist

    def find_grid_index(self, r0_v):
        ng_v = self.ngc_v

        # r0_v cartesian vector in Angstrom
        # assumes self.r_vR being on a regular grid
        # evaluate grid index of cartesian vector
        # convert to reduced (fractional) coordinates
        s0_v = np.linalg.solve(self.cell_prs.T, r0_v)
        return np.array(np.round(ng_v * s0_v, 0), dtype=int) % ng_v

    def bulk_atom_average(self):

        # locate atom farest away from the defect
        rdefect_v = self.atoms_prs.positions[self.defect_index, :]
        bulk_index = np.argmax(self.prs_mic_dist(rdefect_v))

        # return grid indices of region around the bulk atoms
        rbulk_v = self.atoms_prs.positions[bulk_index, :]
        dist = self.grid_mic_dist(rbulk_v)

        # set region as sphere with radius self.ravg
        self.region = np.where(dist < self.ravg)

    @timeit
    def define_averaging_region(self):
        # in pristine obtain atom positions closest to the defect_site
        self.defect_index = np.argmin(self.prs_mic_dist(self.r0))

        self.bulk_atom_average()

    def coarsen_grid(self, nfreq):
        self.phic_prs = self.phi_prs[::nfreq, ::nfreq, ::nfreq]
        self.phic_def = self.phi_def[::nfreq, ::nfreq, ::nfreq]
        self.rc_vR = self.r_vR[:, ::nfreq, ::nfreq, ::nfreq]
        self.ngc_v = np.array(self.rc_vR.shape[1:])

    @timeit
    def calculate_potential_profile(self, nfreq=2, nsample=8):
        self.extract_electrostatic_potentials()
        self.coarsen_grid(nfreq=nfreq)

        # restrict to z-axis
        # use coarse grids
        phiz_prs = np.mean(self.phic_prs, axis=(0, 1))
        phiz_def = np.mean(self.phic_def, axis=(0, 1))

        # get model potential along zaxis
        # expensive for large arrays z_vR
        # therefore coarsen in x-y plane
        # we evaluate on (nsample, nsample, nz) grid
        nx, ny, nz = self.ngc_v
        ix = np.linspace(0, nx - 1, nsample, dtype=int)
        iy = np.linspace(0, ny - 1, nsample, dtype=int)
        igx, igy = np.meshgrid(ix, iy)

        z_vR = self.rc_vR[:, igx, igy, :]
        phi_model = self.calculate_model_potential(z_vR)
        phiz_model = np.mean(phi_model, axis=(0, 1))

        zaxis = self.rc_vR[2, 0, 0, :]
        sz = np.argsort(zaxis)

        dphi = self.calculate_potential_alignment()

        # sorting and conversion to Angstrom
        profile = {'z': zaxis[sz] * Bohr, 'model': phiz_model[sz],
                   'prs': phiz_prs[sz], 'def': phiz_def[sz],
                   'dphi': dphi}

        return profile

    @timeit
    def calculate_potential_alignment(self):
        if self.dphi is not None:
            return self.dphi

        # extract electro-static potential and grid from
        # pristine and defect
        self.extract_electrostatic_potentials()
        self.coarsen_grid(nfreq=self.nfreq)

        # define region away from defect
        self.define_averaging_region()

        # restrict to averaging region
        # use coarse grid
        ix, iy, iz = self.region
        phi_prs = self.phic_prs[ix, iy, iz]
        phi_def = self.phic_def[ix, iy, iz]
        r_vR = self.rc_vR[:, ix, iy, iz]

        # get model potential inside the averaging region
        phi_model = self.calculate_model_potential(r_vR)

        self.dphi = np.average(phi_model - (phi_def - phi_prs))
        return self.dphi

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
