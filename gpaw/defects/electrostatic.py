""" Defects module """

import numpy as np
from gpaw import GPAW, PW
from gpaw.mpi import serial_comm
from ase.units import Bohr, Hartree
from ase.geometry import find_mic


_avg_methods_ = ['atoms', 'sparse-planar', 'full-planar']


def gather_electrostatic_potential(calc):
    if calc.old:
        fine_grid = calc.wfs.pd.gd.refine()
        r_vR = fine_grid.get_grid_point_coordinates(global_array=True)
    else:
        fine_grid = calc.dft.pot_calc.fine_grid
        # make new serial grid descriptor
        r_vR = fine_grid.new(comm=serial_comm).xyz().transpose(3, 0, 1, 2)
    phi = calc.get_electrostatic_potential()
    return r_vR, phi


class ElectrostaticCorrections():
    """
    Calculate the electrostatic corrections for charged defects.

    atoms_prs ... (defect) strucure
    rphi_prs  ... tuple of cartesian finegrid vectors [Bohr]
                  - as obtained from
                  calc.wfs.pd.gd.refine().get_grid_point_coordinates()
                  and corresponding electrostatic potential [eV]
                  - as obtained from
                  calc.get_electrostatic_potential()
                  (r_vR, phi_R)_prs
    rphi_def  ... tuple for defective system
    charge    ... charge state of the defect calculation
    epsilon   ... macroscopic electrostatic constant of the host system
    sigma     ... spread of the Gaussian model charge distribution [Bohr]
    r0        ... defect position [Angstrom]
    ravg      ... average radius for bulk-atom average
    method    ... method selection string
    comm      ... communicator

    """
    def __init__(self, atoms_prs, rphi_prs, rphi_def,
                 charge=None, epsilon=None, sigma=None, r0=None,
                 ravg=2.5, method='full-planar', comm=serial_comm):

        self.atoms_prs = atoms_prs.copy()

        # read and check electrostatic potentials
        self.phi_prs = - rphi_prs[1]
        self.phi_def = - rphi_def[1]
        self.r_vR = rphi_prs[0]       # XXX here: Bohr
        self.ng_v = np.array(self.r_vR.shape[1:])
        r_vR_def = rphi_def[0]

        assert np.allclose(self.r_vR, r_vR_def)
        assert np.allclose(self.phi_prs.shape, self.phi_def.shape)
        assert np.allclose(self.phi_prs.shape, self.ng_v)

        calc = GPAW(mode=PW(500, force_complex_dtype=True),
                    kpts={'size': (1, 1, 1),
                          'gamma': True},
                    parallel={'domain': 1},
                    symmetry='off',
                    communicator=comm,
                    txt=None)

        self.cell_prs = self.atoms_prs.get_cell()
        self.charge = charge
        self.sigma = sigma          # XXX here: Bohr
        self.epsilon = epsilon
        self.r0 = np.array(r0)      # Angstrom
        self.ravg = ravg            # Angstrom

        assert method in _avg_methods_
        self.method = method

        # set grid coarsening
        self.nfreq = 4              # grid coarsening
        if 'full' in method:
            self.nfreq = 1          # no coarsening

        # volume
        self.Omega = self.atoms_prs.get_volume() / Bohr ** 3

        # monoclin check
        self.is_monoclin = np.allclose(self.cell_prs.angles()[:2], [90., 90.])

        # get G vectors
        # init calculator
        calc.initialize(self.atoms_prs)
        pd = calc.wfs.pd
        self.G_Gv = pd.get_reciprocal_vectors(q=0, add_q=False)
        self.G2_G = pd.G2_qG[0]  # |\vec{G}|^2 in Bohr^-2

        # potential alignment
        self.dphi = None

    def calculate_gaussian_density(self):
        # fourier transformed gaussian:
        return self.charge * np.exp(-0.5 * self.G2_G * self.sigma ** 2)

    def calculate_gaussian_potential(self):
        phi_G = np.zeros_like(self.rho_G)
        zero = np.abs(self.G2_G) < 1e-8
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

        G_Gv = self.G_Gv
        phi_G = self.phi_G
        if self.method is not None and 'planar' in self.method:
            # lets try to get away with only G_z vector
            mask_G = (G_Gv[:, 0] == 0) & (G_Gv[:, 1] == 0)
            G_Gv = G_Gv[mask_G]
            phi_G = phi_G[mask_G]

        # assuming G: (ng, 3), r: (3, nx, ny, nz), phi_G: (ng,)
        # compute G * r for all G and grid points
        # shape: (ng, nx, ny, nz)
        Gr = np.einsum('gi,i...->g...', G_Gv, r_vR)

        # compute exp(i * G * r)
        # shape: (ng, nx, ny, nz)
        exp_Gr = np.exp(1j * Gr)

        # weighted sum over G
        # shape: (nx, ny, nz)
        phi_r = np.einsum('g,g...->...', phi_G, exp_Gr)

        assert np.abs(phi_r.imag).max() < 1e-8

        return phi_r.real / self.Omega  # XXX right normalization ?

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
        # in pristine obtain atom positions closest to the defect_site
        defect_index = np.argmin(self.prs_mic_dist(self.r0))

        # locate atom farest away from the defect
        rdefect_v = self.atoms_prs.positions[defect_index, :]
        bulk_index = np.argmax(self.prs_mic_dist(rdefect_v))

        # return grid indices of region around the bulk atoms
        rbulk_v = self.atoms_prs.positions[bulk_index, :]
        dist = self.grid_mic_dist(rbulk_v)

        # set region as sphere with radius self.ravg
        self.region = np.where(dist < self.ravg)


    def planar_average(self, nsample=25, nmin=2):
        # check that monoclinic: z-axis is 90 deg on the plane
        assert self.is_monoclin

        nx, ny, nz = self.ngc_v

        # find defect grid index
        _, _, idef_z = self.find_grid_index(self.r0)
        iblk_z = (idef_z + nz // 2) % nz

        if 'full' in self.method:
            deltan = nz // 8
            nxmax = nx
            nymax = ny
        else:
            deltan = np.min([np.max([nz // 8, nmin]), nsample])
            nxmax = nsample
            nymax = nsample

        ix = np.linspace(0, nx - 1, nxmax, dtype=int)
        iy = np.linspace(0, ny - 1, nymax, dtype=int)
        iz = np.arange(iblk_z - deltan, iblk_z + deltan + 1) % nz

        igx, igy, igz = np.meshgrid(ix, iy, iz)

        self.region = (igx, igy, igz)

    def define_averaging_region(self):
        if self.method == 'atoms':
            # average around "bulk atom"
            print('bulk atom average')
            self.bulk_atom_average()
        else:
            print(f'{self.method} average')
            self.planar_average()

    def coarsen_grid(self, nfreq):
        self.phic_prs = self.phi_prs[::nfreq, ::nfreq, ::nfreq]
        self.phic_def = self.phi_def[::nfreq, ::nfreq, ::nfreq]
        self.rc_vR = self.r_vR[:, ::nfreq, ::nfreq, ::nfreq]
        self.ngc_v = np.array(self.rc_vR.shape[1:])

    def calculate_potential_profile(self, nfreq=2, nsample=8):
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

    def calculate_potential_alignment(self):
        if self.dphi is not None:
            return self.dphi

        # coarsen grid
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

        dphi = phi_model - (phi_def - phi_prs)
        dphi_avg = np.average(dphi)
        # standard deviation of the mean
        dphi_dev = np.std(dphi)

        print('averaging region dphi_avg =', f'{dphi_avg} +- {dphi_dev}')
        self.dphi = dphi_avg

        return self.dphi

    def calculate_correction(self):
        Eli = self.calculate_isolated_correction()
        Elp = self.calculate_periodic_correction()
        Delta_V = self.calculate_potential_alignment()
        print('Eli=', Eli, 'Elp=', Elp, 'Delta_V=', Delta_V)
        return - (Elp - Eli) + Delta_V * self.charge
