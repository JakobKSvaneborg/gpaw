import pickle
import numpy as np
from math import pi
import ase.units
import os

Hartree = ase.units.Hartree
Bohr = ase.units.Bohr


def load(fd):
    try:
        return pickle.load(fd, encoding='latin1')
    except TypeError:
        return pickle.load(fd)


class BuildingBlock:

    """ Module for using Linear response to calculate dielectric
    building block of 2D material with GPAW"""

    def __init__(self, filename, df, isotropic_q=True, nq_inf=10,
                 direction='x', qmax=None, txt='-'):
        """Creates a BuildingBlock object.

        filename: str
            used to save data file: filename-chi.npz
        df: DielectricFunction object
            Determines how linear response calculation is performed
        isotropic_q: bool
            If True, only q-points along one direction (1 0 0) in the
            2D BZ is included, thus assuming an isotropic material
        direction: 'x' or 'y'
            Direction used for isotropic q sampling.
        qmax: float
            Cutoff for q-grid. To be used if one wishes to sample outside the
            irreducible BZ. Only works for isotropic q-sampling.
        nq_inf: int
            number of extra q points in the limit q->0 along each direction,
            extrapolated from q=0, assuming that the head of chi0_wGG goes
            as q^2 and the wings as q.
            Note that this does not hold for (semi)metals!

        """
        assert isotropic_q, "Non-isotropic calculation" \
            + " temporarily turned-off until properly tested."
        if qmax is not None:
            assert isotropic_q
        self.filename = filename
        self.isotropic_q = isotropic_q
        self.nq_inf = nq_inf
        self.nq_inftot = nq_inf
        if not isotropic_q:
            self.nq_inftot *= 2

        if direction == 'x':
            qdir = 0
        elif direction == 'y':
            qdir = 1
        self.direction = direction

        self.df = df  # dielectric function object
        assert self.df.coulomb.truncation == '2D'
        self.wd = self.df.wd

        self.context = self.df.context.with_txt(txt)

        gs = self.df.gs
        kd = gs.kd
        self.kd = kd
        r = gs.gd.get_grid_point_coordinates()
        self.z = r[2, 0, 0, :]

        nw = len(self.wd)
        self.chiM_qw = np.zeros([0, nw])
        self.chiD_qw = np.zeros([0, nw])
        self.drhoM_qz = np.zeros([0, self.z.shape[0]])
        self.drhoD_qz = np.zeros([0, self.z.shape[0]])

        # First: choose all ibzq in 2D BZ
        from ase.dft.kpoints import monkhorst_pack
        from gpaw.kpt_descriptor import KPointDescriptor
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        qd = KPointDescriptor(bzq_qc)
        qd.set_symmetry(gs.atoms, kd.symmetry)
        q_cs = qd.ibzk_kc
        rcell_cv = 2 * pi * np.linalg.inv(gs.gd.cell_cv).T
        if isotropic_q:  # only use q along [1 0 0] or [0 1 0] direction.
            Nk = kd.N_c[qdir]
            qx = np.array(range(1, Nk // 2)) / float(Nk)
            q_cs = np.zeros([Nk // 2 - 1, 3])
            q_cs[:, qdir] = qx
            q = 0
            if qmax is not None:
                qmax *= Bohr
                qmax_v = np.zeros([3])
                qmax_v[qdir] = qmax
                q_c = q_cs[-1]
                q_v = np.dot(q_c, rcell_cv)
                q = (q_v**2).sum()**0.5
                assert Nk % 2 == 0
                i = Nk / 2.0
                while q < qmax:
                    if i == Nk:  # omit BZ edge
                        i += 1
                        continue
                    q_c = np.zeros([3])
                    q_c[qdir] = i / Nk
                    q_cs = np.append(q_cs, q_c[np.newaxis, :], axis=0)
                    q_v = np.dot(q_c, rcell_cv)
                    q = (q_v**2).sum()**0.5
                    i += 1
        q_vs = np.dot(q_cs, rcell_cv)
        q_abs = (q_vs**2).sum(axis=1)**0.5
        sort = np.argsort(q_abs)
        q_abs = q_abs[sort]
        q_cs = q_cs[sort]
        if isotropic_q:
            q_cut = q_abs[0] / 2  # extrapolate to half of smallest finite q
        else:
            q_cut = q_abs[1]  # smallest finite q
        self.nq_cut = self.nq_inftot + 1

        q_infs = np.zeros([q_cs.shape[0] + self.nq_inftot, 3])
        # x-direction:
        q_infs[: self.nq_inftot, qdir] = \
            np.linspace(1e-05, q_cut, self.nq_inftot + 1)[:-1]
        if not isotropic_q:  # y-direction
            q_infs[self.nq_inf:self.nq_inftot, 1] = \
                np.linspace(0, q_cut, self.nq_inf + 1)[1:]

        # add q_inf to list
        self.q_cs = np.insert(q_cs, 0, np.zeros([self.nq_inftot, 3]), axis=0)
        self.q_vs = np.dot(self.q_cs, rcell_cv)
        self.q_vs += q_infs
        self.q_abs = (self.q_vs**2).sum(axis=1)**0.5
        self.q_infs = q_infs
        self.complete = False
        self.last_q_idx = 0
        if self.load_chi_file():
            if self.complete:
                self.context.print('Building block loaded from file')
        self.context.comm.barrier()

    def calculate_building_block(self, add_intraband=False):
        if self.complete:
            return
        Nq = self.q_cs.shape[0]
        for current_q_idx in range(self.last_q_idx, Nq):
            self.save_chi_file(q_idx=current_q_idx)
            self.last_q_idx = current_q_idx
            q_c = self.q_cs[current_q_idx]
            q_inf = self.q_infs[current_q_idx]
            if np.allclose(q_inf, 0):
                q_inf = None

            qcstr = '(' + ', '.join(['%.3f' % x for x in q_c]) + ')'
            self.context.print(
                'Calculating contribution from q-point #%d/%d, q_c=%s' % (
                    current_q_idx + 1, Nq, qcstr), flush=False)
            if q_inf is not None:
                qstr = '(' + ', '.join(['%.3f' % x for x in q_inf]) + ')'
                self.context.print('    and q_inf=%s' % qstr, flush=False)
            qpd, chi0_wGG, \
                chi_wGG = self.df.get_dielectric_matrix(
                    symmetric=False,
                    calculate_chi=True,
                    q_c=q_c,
                    q_v=q_inf,
                    direction=self.direction,
                    add_intraband=add_intraband)
            self.context.print('calculated chi!')

            nw = len(self.wd)
            comm = self.context.comm
            w1 = min(self.df.blocks1d.blocksize * comm.rank, nw)

            chiM_w, chiD_w, drhoM_z, drhoD_z = \
                self.get_chi_2D(qpd, chi_wGG)
            chiM_w = self.collect(chiM_w)
            chiD_w = self.collect(chiD_w)

            if self.context.comm.rank == 0:
                assert w1 == 0  # drhoM and drhoD in static limit
                self.update_building_block(chiM_w[np.newaxis, :],
                                           chiD_w[np.newaxis, :],
                                           drhoM_z[np.newaxis, :],
                                           drhoD_z[np.newaxis, :])

        # Induced densities are not properly described in q-> 0 limit-
        # replace with finite q result:
        if self.context.comm.rank == 0:
            for n in range(Nq):
                if np.allclose(self.q_cs[n], 0):
                    self.drhoM_qz[n] = self.drhoM_qz[self.nq_cut]
                    self.drhoD_qz[n] = self.drhoD_qz[self.nq_cut]

        self.complete = True
        self.save_chi_file()

        return

    def update_building_block(self, chiM_qw, chiD_qw, drhoM_qz,
                              drhoD_qz):

        self.chiM_qw = np.append(self.chiM_qw, chiM_qw, axis=0)
        self.chiD_qw = np.append(self.chiD_qw, chiD_qw, axis=0)
        self.drhoM_qz = np.append(self.drhoM_qz, drhoM_qz, axis=0)
        self.drhoD_qz = np.append(self.drhoD_qz, drhoD_qz, axis=0)

    def get_chi_2D(self, qpd, chi_wGG):
        r"""Calculate the monopole and dipole contribution to the
        2D susceptibility chi_2D for single q-point q, defined as

        ::

          \chi^M_2D(q, \omega) = \int\int dr dr' \chi(q, \omega, r,r') \\
                              = L \chi_{G=G'=0}(q, \omega)
          \chi^D_2D(q, \omega) = \int\int dr dr' z \chi(q, \omega, r,r') z'
                               = 1/L sum_{G_z,G_z'} z_factor(G_z)
                               chi_{G_z,G_z'} z_factor(G_z'),
          Where z_factor(G_z) =  +/- i e^{+/- i*G_z*z0}
          (L G_z cos(G_z L/2)-2 sin(G_z L/2))/G_z^2

        qpd: Single q-point descriptor
        chi_wGG: Susceptibility in PW basis
          """

        nw = chi_wGG.shape[0]
        z = self.z
        L = qpd.gd.cell_cv[2, 2]  # Length of cell in Bohr

        # XXX This seems like a bit dangerous assumption
        z0 = L / 2.  # position of layer
        chiM_w = np.zeros([nw], dtype=complex)
        chiD_w = np.zeros([nw], dtype=complex)
        drhoM_z = np.zeros([len(z)], dtype=complex)  # induced density
        drhoD_z = np.zeros([len(z)], dtype=complex)  # induced dipole density

        npw = chi_wGG.shape[1]
        G_Gv = qpd.get_reciprocal_vectors(add_q=False)

        Glist = []
        for iG in range(npw):  # List of G with Gx,Gy = 0
            if G_Gv[iG, 0] == 0 and G_Gv[iG, 1] == 0:
                Glist.append(iG)

        # If node lacks frequency points due to block parallelization then
        # return empty arrays
        if nw == 0:
            return chiM_w, chiD_w, drhoM_z, drhoD_z
        chiM_w = L * chi_wGG[:, 0, 0]
        drhoM_z += chi_wGG[0, 0, 0]
        for iG in Glist[1:]:
            G_z = G_Gv[iG, 2]
            qGr_R = np.inner(G_z, z.T).T
            # Fourier transform to get induced density at \omega=0
            drhoM_z += np.exp(1j * qGr_R) * chi_wGG[0, iG, 0]
            for iG1 in Glist[1:]:
                G_z1 = G_Gv[iG1, 2]
                # integrate with z along both coordinates
                factor = z_factor(z0, L, G_z)
                factor1 = z_factor(z0, L, G_z1, sign=-1)
                chiD_w[:] += 1. / L * factor * chi_wGG[:, iG, iG1] * \
                    factor1
                # induced dipole density due to V_ext = z
                drhoD_z[:] += 1. / L * np.exp(1j * qGr_R) * \
                    chi_wGG[0, iG, iG1] * factor1
        # Normalize induced densities with chi
        if nw != 0:
            drhoM_z /= chiM_w[0]
            drhoD_z /= chiD_w[0]

        """ Returns chi2D monopole and dipole, induced
        densities and z array (all in Bohr)
        """
        return chiM_w, chiD_w, drhoM_z, drhoD_z

    def save_chi_file(self, filename=None, q_idx=None):
        if q_idx is None:
            q_idx = self.last_q_idx
        if filename is None:
            filename = self.filename
        data = {'last_q': q_idx,
                'complete': self.complete,
                'isotropic_q': self.isotropic_q,
                'q_cs': self.q_cs,
                'q_vs': self.q_vs,
                'q_abs': self.q_abs,
                'omega_w': self.wd.omega_w,
                'chiM_qw': self.chiM_qw,
                'chiD_qw': self.chiD_qw,
                'z': self.z,
                'drhoM_qz': self.drhoM_qz,
                'drhoD_qz': self.drhoD_qz}

        if self.context.comm.rank == 0:
            np.savez_compressed(filename + '-chi.npz',
                                **data)
        self.context.comm.barrier()

    def load_chi_file(self):
        try:
            data = np.load(self.filename + '-chi.npz')
        except OSError:
            return False
        if (np.all(data['omega_w'] == self.wd.omega_w) and
            np.all(data['q_cs'] == self.q_cs) and
            np.all(data['z'] == self.z)):
            self.last_q_idx = data['last_q']
            self.complete = data['complete']
            self.chiM_qw = data['chiM_qw']
            self.chiD_qw = data['chiD_qw']
            self.drhoM_qz = data['drhoM_qz']
            self.drhoD_qz = data['drhoD_qz']
            return True
        else:
            return False

    def interpolate_to_grid(self, q_grid, w_grid):

        """
        Parameters
        q_grid: in Ang. should start at q=0
        w_grid: in eV
        """

        from scipy.interpolate import RectBivariateSpline
        from scipy.interpolate import interp1d
        from gpaw.response.frequencies import FrequencyGridDescriptor
        if not self.complete:
            self.calculate_building_block()
        q_grid *= Bohr
        w_grid /= Hartree

        assert np.max(q_grid) <= np.max(self.q_abs), \
            'q can not be larger that %1.2f Ang' % np.max(self.q_abs / Bohr)
        assert np.max(w_grid) <= np.max(self.wd.omega_w), \
            'w can not be larger that %1.2f eV' % \
            np.max(self.wd.omega_w * Hartree)

        sort = np.argsort(self.q_abs)
        q_abs = self.q_abs[sort]

        # chi monopole
        self.chiM_qw = self.chiM_qw[sort]

        omit_q0 = False
        if np.isclose(q_abs[0], 0) and not np.isclose(self.chiM_qw[0, 0], 0):
            omit_q0 = True  # omit q=0 from interpolation
            q0_abs = q_abs[0].copy()
            q_abs[0] = 0.
            chi0_w = self.chiM_qw[0].copy()
            self.chiM_qw[0] = np.zeros_like(chi0_w)

        yr = RectBivariateSpline(q_abs, self.wd.omega_w,
                                 self.chiM_qw.real,
                                 s=0)

        yi = RectBivariateSpline(q_abs, self.wd.omega_w,
                                 self.chiM_qw.imag, s=0)

        self.chiM_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)
        if omit_q0:
            yr = interp1d(self.wd.omega_w, chi0_w.real)
            yi = interp1d(self.wd.omega_w, chi0_w.imag)
            chi0_w = yr(w_grid) + 1j * yi(w_grid)
            q_abs[0] = q0_abs
            if np.isclose(q_grid[0], 0):
                self.chiM_qw[0] = chi0_w

        # chi dipole
        yr = RectBivariateSpline(q_abs, self.wd.omega_w,
                                 self.chiD_qw[sort].real,
                                 s=0)
        yi = RectBivariateSpline(q_abs, self.wd.omega_w,
                                 self.chiD_qw[sort].imag,
                                 s=0)

        self.chiD_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)

        # drho monopole

        yr = RectBivariateSpline(q_abs, self.z,
                                 self.drhoM_qz[sort].real, s=0)
        yi = RectBivariateSpline(q_abs, self.z,
                                 self.drhoM_qz[sort].imag, s=0)

        self.drhoM_qz = yr(q_grid, self.z) + 1j * yi(q_grid, self.z)

        # drho dipole
        yr = RectBivariateSpline(q_abs, self.z,
                                 self.drhoD_qz[sort].real, s=0)
        yi = RectBivariateSpline(q_abs, self.z,
                                 self.drhoD_qz[sort].imag, s=0)

        self.drhoD_qz = yr(q_grid, self.z) + 1j * yi(q_grid, self.z)

        self.q_abs = q_grid
        self.wd = FrequencyGridDescriptor(w_grid)
        self.save_chi_file(filename=self.filename + '_int')

    def collect(self, a_w):
        comm = self.context.comm
        mynw = self.df.blocks1d.blocksize
        b_w = np.zeros(mynw, a_w.dtype)
        b_w[:self.df.blocks1d.nlocal] = a_w
        nw = len(self.wd)
        A_w = np.empty(comm.size * mynw, a_w.dtype)
        comm.all_gather(b_w, A_w)
        return A_w[:nw]

    def clear_temp_files(self):
        if not self.savechi0:
            comm = self.context.comm
            if comm.rank == 0:
                while len(self.temp_files) > 0:
                    filename = self.temp_files.pop()
                    os.remove(filename)


"""TOOLS"""


def check_building_blocks(BBfiles=None):
    """ Check that building blocks are on same frequency-
    and q- grid.

    BBfiles: list of str
        list of names of BB files
    """
    name = BBfiles[0] + '-chi.npz'
    data = np.load(name)
    try:
        q = data['q_abs'].copy()
        w = data['omega_w'].copy()
    except TypeError:
        # Skip test for old format:
        return True
    for name in BBfiles[1:]:
        data = np.load(name + '-chi.npz')
        if len(w) != len(data['omega_w']):
            return False
        elif not ((data['q_abs'] == q).all() and
                  (data['omega_w'] == w).all()):
            return False
    return True


def z_factor(z0, d, G, sign=1):
    factor = -1j * sign * np.exp(1j * sign * G * z0) * \
        (d * G * np.cos(G * d / 2.) - 2. * np.sin(G * d / 2.)) / G**2
    return factor


def z_factor2(z0, d, G, sign=1):
    factor = sign * np.exp(1j * sign * G * z0) * np.sin(G * d / 2.)
    return factor


def expand_layers(structure):
    newlist = []
    for name in structure:
        num = ''
        while name[0].isdigit():
            num += name[0]
            name = name[1:]
        try:
            num = int(num)
        except ValueError:
            num = 1
        for n in range(num):
            newlist.append(name)
    return newlist


def read_chi_wGG(name):
    """
    Read density response matrix calculated with the DielectricFunction
    module in GPAW.
    Returns frequency grid, gpaw.wavefunctions object, chi_wGG
    """
    fd = open(name, 'rb')
    omega_w, qpd, chi_wGG, q0, chi0_wvv = load(fd)
    nw = len(omega_w)
    nG = qpd.ngmax
    chi_wGG = np.empty((nw, nG, nG), complex)
    for chi_GG in chi_wGG:
        chi_GG[:] = load(fd)
    return omega_w, qpd, chi_wGG, q0
