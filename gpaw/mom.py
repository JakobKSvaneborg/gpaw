"""Module for calculations using the Maximum Overlap Method (MOM).

See:

   https://arxiv.org/abs/2102.06542,
   :doi:`10.1021/acs.jctc.0c00597`.
"""

import numpy as np

from ase.units import Ha

from gpaw.occupations import FixedOccupationNumbers, ParallelLayout
from scipy.optimize import linear_sum_assignment


def prepare_mom_calculation(calc,
                            atoms,
                            numbers,
                            use_projections=True,
                            update_numbers=True,
                            use_fixed_occupations=False,
                            width=0.0,
                            niter_width_update=40,
                            width_increment=0.0):
    """Helper function to prepare a calculator for a MOM calculation.

    calc: GPAW instance
        GPAW calculator object.
    atoms: ASE instance
        ASE atoms object.
    numbers: list (len=nspins) of lists (len=nbands)
        Occupation numbers (in the range from 0 to 1). Used
        for the initialization of the MOM reference orbitals.
    use_projections: bool
        If True (default), the occupied orbitals at iteration k are
        chosen as the orbitals ``|psi^(k)_m>`` with the biggest
        weights ``P_m`` evaluated as the projections onto the manifold
        of reference orbitals ``|psi_n>``: ``P_m = (Sum_n(|O_nm|^2))^0.5
        (O_nm = <psi_n|psi^(k)_m>)`` see :doi:`10.1021/acs.jctc.7b00994`.
        If False, the weights are evaluated as: ``P_m = max_n(|O_nm|)``,
        see :doi:`10.1021/acs.jctc.0c00488`.
    update_numbers: bool
        If True (default), 'numbers' gets updated with the calculated
        occupation numbers, and when changing atomic positions
        the MOM reference orbitals will be initialized as the
        occupied orbitals found at convergence for the previous
        geometry. If False, when changing positions the MOM
        reference orbitals will be initialized as the orbitals
        of the previous geometry corresponding to the user-supplied
        'numbers'.
    use_fixed_occupations: bool
        If False (default), the MOM algorithm is used. If True,
        fixed occupations will be used.
    width: float
        Width of Gaussian function in eV for smearing of holes
        and excited electrons. The holes and excited electrons
        are found with respect to the zero-width ground-state
        occupations.
        See :doi:`10.1021/acs.jctc.0c00597`.
    niter_width_update: int
        Number of iterations after which the width of the
        Gaussian smearing function is increased.
    width_increment: float
        How much to increase the width of the Gaussian smearing
        function.
    """

    new_gpaw = hasattr(calc, '_dft')
    if not new_gpaw:
        if calc.wfs is None:
            # We need the wfs object to initialize OccupationsMOM
            calc.initialize(atoms)
    else:
        if calc._dft is None:
            calc.create_new_calculation(atoms)

    occ_mom = OccupationsMOM(calc.wfs,
                             numbers,
                             use_projections,
                             update_numbers,
                             use_fixed_occupations,
                             width,
                             niter_width_update,
                             width_increment)
    if not new_gpaw:
        calc.set(occupations=occ_mom, _set_ok=True)
    else:
        calc.dft.scf_loop.occ_calc.occ = occ_mom
        calc.dft.results = {}

    calc.log(occ_mom)

    return occ_mom


class OccupationsMOM:
    """MOM occupation class.

    The occupation numbers are found using a maximum overlap
    criterion and then broadcasted in occupations.py using the
    _calculate method of FixedOccupationNumbers.
    """

    def __init__(self,
                 wfs,
                 numbers,
                 use_projections=False,
                 update_numbers=True,
                 use_fixed_occupations=False,
                 width=0.0,
                 niter_width_update=10,
                 width_increment=0.0):
        self.wfs = wfs
        self.numbers = np.array(numbers)
        self.use_projections = use_projections
        self.update_numbers = update_numbers
        self.use_fixed_occupations = use_fixed_occupations
        self.width = width / Ha
        self.niter_width_update = niter_width_update
        self.width_increment = width_increment / Ha

        parallel_layout = ParallelLayout(self.wfs.bd,
                                         self.wfs.kd.comm,
                                         self.wfs.gd.comm)
        self.occ = FixedOccupationNumbers(numbers, parallel_layout)
        self.extrapolate_factor = self.occ.extrapolate_factor

        self.name = 'mom'
        self.iters = 0
        self.initialized = False

    def todict(self):
        dct = {'name': self.name,
               'numbers': self.numbers,
               'use_projections': self.use_projections,
               'update_numbers': self.update_numbers,
               'use_fixed_occupations': self.use_fixed_occupations}
        if self.width != 0.0:
            dct['width'] = self.width * Ha
            dct['niter_width_update'] = self.niter_width_update
            dct['width_increment'] = self.width_increment * Ha
        return dct

    def __str__(self):
        s = 'Excited-state calculation with Maximum Overlap Method\n'
        s += '  Gaussian smearing of holes and excited electrons: '
        if self.width == 0.0:
            s += 'off\n'
        else:
            s += f'{self.width * Ha:.4f} eV\n'
        return s

    def calculate(self,
                  nelectrons,
                  eigenvalues,
                  weights,
                  fermi_levels_guess,
                  fix_fermi_level=False):
        assert not fix_fermi_level

        if not self.initialized:
            # If MOM reference orbitals are not initialized yet (e.g. when
            # the calculation is initialized from atomic densities), update
            # the occupation numbers according to the user-supplied 'numbers'
            self.occ.f_sn = self.numbers.copy()
            self.initialize_reference_orbitals()
        else:
            self.occ.f_sn = self.update_occupations()
            self.iters += 1

        f_qn, fermi_levels, e_entropy = self.occ.calculate(nelectrons,
                                                           eigenvalues,
                                                           weights,
                                                           fermi_levels_guess)
        return f_qn, fermi_levels, e_entropy

    def initialize_reference_orbitals(self):
        try:
            f_n = self.wfs.kpt_u[0].f_n
        except ValueError:  # new gpaw
            return
        if f_n is None:  # old gpaw
            # If the occupation numbers are not already available
            # (e.g. when the calculation is initialized from atomic
            # densities) we first need to take a step of eigensolver
            # and update the occupation numbers according to the
            # 'user-supplied' numbers before initializing the MOM
            # reference orbitals
            return

        self.iters = 0

        if self.use_fixed_occupations:
            self.initialized = True
            return

        # Initialize MOM reference orbitals for each equally
        # occupied subspace separately
        self.subspace_mask = self.find_unique_occupation_numbers()
        if self.wfs.mode == 'lcao':
            self.c_ref = {}
            for kpt in self.wfs.kpt_u:
                self.c_ref[kpt.s] = {}
                for f_n_unique in self.subspace_mask[kpt.s]:
                    occupied = self.subspace_mask[kpt.s][f_n_unique]
                    self.c_ref[kpt.s][f_n_unique] = kpt.C_nM[occupied].copy()
        else:
            self.wf = {}
            self.p_an = {}
            for kpt in self.wfs.kpt_u:
                self.wf[kpt.s] = {}
                self.p_an[kpt.s] = {}
                for f_n_unique in self.subspace_mask[kpt.s]:
                    occupied = self.subspace_mask[kpt.s][f_n_unique]
                    # Pseudo wave functions
                    self.wf[kpt.s][f_n_unique] = kpt.psit_nG[occupied].copy()
                    # Atomic contributions times projector overlaps
                    self.p_an[kpt.s][f_n_unique] = \
                        {a: np.dot(self.wfs.setups[a].dO_ii, P_ni[occupied].T)
                         for a, P_ni in kpt.P_ani.items()}

        self.initialized = True

    def update_occupations(self):
        if self.width != 0.0:
            if self.iters == 0:
                self.width_update_counter = 0
            if self.iters % self.niter_width_update == 0:
                self.gauss_width = self.width + self.width_update_counter \
                    * self.width_increment
                self.width_update_counter += 1

        if not self.use_fixed_occupations:
            f_sn = np.zeros_like(self.numbers)
            for kpt in self.wfs.kpt_u:
                # Compute weights with respect to each equally occupied
                # subspace of the reference orbitals and occupy orbitals
                # with biggest weights

                # number of subspaces
                nsubs = len(self.subspace_mask[kpt.s])
                if nsubs == 1:
                    # we can just occupy the orbitals
                    # with largest projections
                    f_n_unique = list(self.subspace_mask[kpt.s].keys())[0]
                    subspace_mask = self.subspace_mask[kpt.s][f_n_unique]
                    sub_size = np.sum(subspace_mask)
                    P_m = self.calculate_weights(kpt, f_n_unique)

                    # we could use np.argpartition here
                    # but argsort is better readable
                    occ_mask = np.argsort(P_m)[::-1][:sub_size]
                    f_sn[kpt.s][occ_mask] = f_n_unique
                else:
                    # for more than one subspace we need to figure out the
                    # assignment with the maximum projections

                    nband = len(self.numbers[kpt.s])
                    Ps_m = np.zeros((nband, nband))
                    fs_key = []
                    sidx = 0
                    for ss, f_n_unique in enumerate(self.subspace_mask[kpt.s]):
                        subspace_mask = self.subspace_mask[kpt.s][f_n_unique]
                        sub_size = np.sum(subspace_mask)
                        # Ps_m ... projections of the subspace orbitals
                        # to the scf orbitals from the k-iteration
                        w = self.calculate_weights(kpt, f_n_unique)
                        Ps_m[sidx: sidx + sub_size, :] = w[None, :]
                        fs_key += sub_size * [f_n_unique]
                        sidx += sub_size

                    Ps_m = Ps_m[:sidx, :]
                    # Ps_m.shape = noccupied, nbands
                    noccupied = np.sum(self.numbers[kpt.s] > 1.0e-10)
                    assert (Ps_m.shape[0] == noccupied)

                    # Optimize assigment of subspace occupations
                    # such that sum of the selected projections is maximal
                    row_ind, col_ind = linear_sum_assignment(-Ps_m)

                    # select the subspace index from rol_ind
                    # assign band index (=col_ind) with occupation number
                    for irow, icol in zip(row_ind, col_ind):
                        f_n_unique = fs_key[irow]
                        f_sn[kpt.s][icol] = f_n_unique

                if self.update_numbers:
                    self.numbers[kpt.s] = f_sn[kpt.s].copy()
        else:
            f_sn = self.numbers.copy()

        for kpt in self.wfs.kpt_u:
            if self.width != 0.0:
                orbs, f_sn_gs = self.find_hole_and_excited_orbitals(f_sn, kpt)
                if orbs:
                    for o in orbs:
                        mask, gauss = self.gaussian_smearing(kpt,
                                                             f_sn_gs,
                                                             o,
                                                             self.gauss_width)
                        f_sn_gs[mask] += (o[1] * gauss)
                    f_sn[kpt.s] = f_sn_gs.copy()

        return f_sn

    def calculate_weights(self, kpt, f_n_unique):
        if self.wfs.mode == 'lcao':
            O = np.dot(self.c_ref[kpt.s][f_n_unique].conj(),
                       np.dot(kpt.S_MM, kpt.C_nM[:].T))
        else:
            # Pseudo wave function overlaps
            O = self.wfs.integrate(self.wf[kpt.s][f_n_unique][:],
                                   kpt.psit_nG[:][:], True)

            # Atomic contributions
            O_corr = np.zeros_like(O)
            for a, p_a in self.p_an[kpt.s][f_n_unique].items():
                O_corr += np.dot(kpt.P_ani[a][:].conj(), p_a).T
            O_corr = np.ascontiguousarray(O_corr)
            self.wfs.gd.comm.sum(O_corr)

            # Sum pseudo wave and atomic contributions
            O += O_corr

        if self.use_projections:
            P = np.sum(abs(O)**2, axis=0)
            P = P**0.5
        else:
            P = np.amax(abs(O), axis=0)

        return P

    def find_hole_and_excited_orbitals(self, f_sn, kpt):
        # Zero-width occupations for ground state
        ne = int(f_sn[kpt.s].sum())
        f_sn_gs = np.zeros_like(f_sn[kpt.s])
        f_sn_gs[:ne] = 1.0
        f_sn_diff = f_sn[kpt.s] - f_sn_gs

        # Select hole and excited orbitals
        idxs = np.where(np.abs(f_sn_diff) > 1e-5)[0]
        w = f_sn_diff[np.abs(f_sn_diff) > 1e-5]
        orbs = list(zip(idxs, w))

        return orbs, f_sn_gs

    def gaussian_smearing(self, kpt, f_sn_gs, o, gauss_width):
        if o[1] < 0:
            mask = (f_sn_gs > 1e-8)
        else:
            mask = (f_sn_gs < 1e-8)

        e = kpt.eps_n[mask]
        de2 = -(e - kpt.eps_n[o[0]]) ** 2
        gauss = (1 / (gauss_width * np.sqrt(2 * np.pi)) *
                 np.exp(de2 / (2 * gauss_width ** 2)))
        gauss /= sum(gauss)

        return mask, gauss

    def find_unique_occupation_numbers(self):
        f_sn_unique = {}
        for kpt in self.wfs.kpt_u:
            f_sn_unique[kpt.s] = {}
            f_n = self.numbers[kpt.s]

            for f_n_unique in np.unique(f_n):
                if f_n_unique >= 1.0e-10:
                    f_sn_unique[kpt.s][f_n_unique] = f_n == f_n_unique

        return f_sn_unique
