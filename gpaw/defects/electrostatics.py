""" Defects module """

import numpy as np
from gpaw.mpi import serial_comm
from gpaw.core import PWDesc, UGDesc, UGArray
from ase.units import Bohr, Hartree
from ase.geometry import find_mic


_avg_methods_ = ['atoms', 'sparse-planar', 'full-planar']


def build_ugarray(atoms, data):
    grid = UGDesc(cell=atoms.cell, size=data.shape, pbc=atoms.pbc)
    return UGArray(grid, data=data)


def gather_electrostatic_potential(calc):
    if calc.old:
        # create UGArray from old GPAW data
        phi_r = calc.get_electrostatic_potential()
        atoms = calc.get_atoms()
        phi_R = build_ugarray(atoms, phi_r)
    else:
        phi_R = calc.dft.electrostatic_potential().pseudo_potential()

    # XXX should get rid of broadcast
    return phi_R.gather(broadcast=True)


def plot_potentials(profile, png=None):
    from matplotlib import pyplot as plt

    z = profile['z']
    V_m = profile['model']
    dV_defprs = profile['def'] - profile['prs']
    dV = V_m - dV_defprs
    dphi_avg = profile['dphi']

    plt.plot(z, dV, '-', label=r'$\Delta V(z)$')
    plt.plot(z, V_m, '-', label='$V(z)$')
    plt.plot(z, dV_defprs, '-',
             label=(r'$[V^{V_\mathrm{Ga}^{-3}}_\mathrm{el}(z) -'
                    r'V^{0}_\mathrm{el}(z) ]$'))

    plt.axhline(dphi_avg, ls='dashed')
    plt.axhline(0.0, ls='-', color='grey')
    plt.xlabel(r'$z\enspace (\mathrm{\AA})$', fontsize=18)
    plt.ylabel('Planar averages (eV)', fontsize=18)
    plt.legend(loc='upper right')
    plt.xlim((z[0], z[-1]))

    if png is not None:
        plt.savefig(png, bbox_inches='tight', dpi=300)
    else:
        plt.show()


class ElectrostaticCorrections():
    """
    Calculate the electrostatic corrections for charged defects.

    phi_pristine  ... UGArray electrostatic_potential pristine [eV]
    phi_defect    ... UGArray electrostatic_potential defect [eV]
    charge    ... charge state of the defect calculation
    epsilon   ... macroscopic electrostatic constant of the host system
    sigma     ... spread of the Gaussian model charge distribution [Bohr]
    r0        ... defect position [Angstrom]
    ravg      ... average radius for bulk-atom average
    method    ... method selection string

    """
    def __init__(self, phi_pristine, phi_defect, ecut=500,
                 charge=None, epsilon=None, sigma=None, r0=None,
                 ravg=2.5, method='full-planar', atoms_pristine=None):

        # read and check electrostatic potentials
        # conversion to Hartree and Bohr
        self.phi_prs = - phi_pristine.data / Hartree    # Hartree
        self.phi_def = - phi_defect.data / Hartree      # Hartree
        r_vR = phi_pristine.desc.xyz().transpose(3, 0, 1, 2)
        self.r_vR = r_vR / Bohr                         # Bohr
        self.ng_v = np.array(r_vR.shape[1:])

        assert np.allclose(self.ng_v, self.phi_def.shape)
        assert np.allclose(self.phi_prs.shape, self.phi_def.shape)

        self.cell_cv = phi_pristine.desc.cell / Bohr    # Bohr
        self.sigma = sigma                              # Bohr
        self.r0 = np.array(r0) / Bohr                   # Bohr
        self.ravg = ravg / Bohr                         # Bohr
        self.charge = charge
        self.epsilon = epsilon

        assert method in _avg_methods_
        self.method = method
        self.atoms_prs = atoms_pristine

        # set grid coarsening
        self.nfreq = 4              # grid coarsening
        if 'full' in method:
            self.nfreq = 1          # no coarsening
        if 'atoms' in method:
            assert self.atoms_prs is not None

        # volume Bohr^3
        self.Omega = np.linalg.det(self.cell_cv)    # Bohr^3

        # monoclin check
        cross_ab = np.cross(self.cell_cv[0, :], self.cell_cv[1, :])
        cross_ab = cross_ab / np.linalg.norm(cross_ab)
        norm_c = self.cell_cv[2, :] / np.linalg.norm(self.cell_cv[2, :])
        self.is_monoclin = np.abs(np.dot(cross_ab, norm_c) - 1.) < 1e-6

        # get G vectors, cut-off in Hartree
        pw_desc = PWDesc(cell=self.cell_cv, ecut=ecut / Hartree,
                         comm=serial_comm, dtype=complex)
        self.G_Gv = pw_desc.reciprocal_vectors()        # Bohr^-1 ?
        # G2
        self.G2_G = np.sum(np.abs(self.G_Gv) ** 2, axis=-1)

        # potential alignment
        self.dphi = None

    def calculate_gaussian_density(self):
        # fourier transformed gaussian:
        return self.charge * np.exp(-0.5 * self.G2_G * self.sigma ** 2)

    def calculate_gaussian_potential(self):
        phi_G = np.zeros_like(self.rho_G)
        zero = np.abs(self.G2_G) < 1e-8
        phi_G[~zero] = self.rho_G[~zero] / self.G2_G[~zero]
        return 4. * np.pi * phi_G / self.epsilon

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
        Eli = 0.5 * self.charge ** 2 / np.pi ** 0.5 / eps / sgm
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

    def prs_mic_dist(self, r0_v):

        atoms = self.atoms_prs
        dR = atoms.positions / Bohr - r0_v[None, :]
        _, dist = find_mic(dR, self.cell_cv)

        return dist

    def grid_mic_dist(self, r_v):
        # coarse grid
        grid_shape = self.ngc_v

        dR = self.rc_vR.T - r_v[None, None, None, :]
        # flatten grid and reshape
        dR = dR.reshape((np.prod(grid_shape), 3))
        _, dist = find_mic(dR, self.cell_cv)
        dist = dist.reshape(grid_shape)

        return dist

    def find_grid_index(self, r0_v):
        ng_v = self.ngc_v

        # r0_v cartesian vector in Bohr
        # assumes self.r_vR being on a regular grid
        # evaluate grid index of cartesian vector
        # convert to reduced (fractional) coordinates
        s0_v = np.linalg.solve(self.cell_cv.T, r0_v)
        return np.array(np.round(ng_v * s0_v, 0), dtype=int) % ng_v

    def bulk_atom_average(self):
        # in pristine obtain atom positions closest to the defect_site
        defect_index = np.argmin(self.prs_mic_dist(self.r0))

        # locate atom farest away from the defect [Bohr]
        rdefect_v = self.atoms_prs.positions[defect_index, :] / Bohr
        bulk_index = np.argmax(self.prs_mic_dist(rdefect_v))

        # return grid indices of region around the bulk atoms [Bohr]
        rbulk_v = self.atoms_prs.positions[bulk_index, :] / Bohr
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

        # sorting and conversion to Angstrom and eV
        profile = {'z': zaxis[sz] * Bohr,
                   'model': phiz_model[sz] * Hartree,
                   'prs': phiz_prs[sz] * Hartree,
                   'def': phiz_def[sz] * Hartree,
                   'dphi': dphi * Hartree}

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

        print('averaging region dphi_avg =',
              f'{dphi_avg * Hartree} +- {dphi_dev * Hartree}')
        self.dphi = dphi_avg

        return self.dphi

    def calculate_correction(self):
        # conversion to eV
        Eli = self.calculate_isolated_correction() * Hartree
        Elp = self.calculate_periodic_correction() * Hartree
        Delta_V = self.calculate_potential_alignment() * Hartree
        print('Eli=', Eli, 'Elp=', Elp, 'Delta_V=', Delta_V)
        return - (Elp - Eli) + Delta_V * self.charge
