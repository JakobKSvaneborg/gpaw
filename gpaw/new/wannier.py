def get_wannier_localization_matrix(dft, nbands, dirG, kpoint,
                                    nextkpoint, G_I, spin):
    """Calculate integrals for maximally localized Wannier functions."""

    # Due to orthorhombic cells, only one component of dirG is non-zero.
    k_kc = self.wfs.kd.bzk_kc
    G_c = k_kc[nextkpoint] - k_kc[kpoint] - G_I

    return self.get_wannier_integrals(spin, kpoint,
                                      nextkpoint, G_c, nbands)


def get_wannier_integrals(self, s, k, k1, G_c, nbands=None):
    """Calculate integrals for maximally localized Wannier functions."""

    assert s <= self.wfs.nspins
    kpt_rank, u = divmod(k + len(self.wfs.kd.ibzk_kc) * s,
                         len(self.wfs.kpt_u))
    kpt_rank1, u1 = divmod(k1 + len(self.wfs.kd.ibzk_kc) * s,
                           len(self.wfs.kpt_u))

    # XXX not for the kpoint/spin parallel case
    assert self.wfs.kd.comm.size == 1

    # If calc is a save file, read in tar references to memory
    # For lcao mode just initialize the wavefunctions from the
    # calculated lcao coefficients
    if self.wfs.mode == 'lcao':
        self.wfs.initialize_wave_functions_from_lcao()
    else:
        self.wfs.initialize_wave_functions_from_restart_file()

    # Get pseudo part
    psit_nR = self.get_realspace_wfs(u)
    psit1_nR = self.get_realspace_wfs(u1)
    Z_nn = self.wfs.gd.wannier_matrix(psit_nR, psit1_nR, G_c, nbands)
    # Add corrections
    self.add_wannier_correction(Z_nn, G_c, u, u1, nbands)

    self.wfs.gd.comm.sum(Z_nn)

    return Z_nn


def add_wannier_correction(self, Z_nn, G_c, u, u1, nbands=None):
    r"""Calculate the correction to the wannier integrals.

    See: (Eq. 27 ref1)::

                      -i G.r
        Z   = <psi | e      |psi >
         nm       n             m

                       __                __
               ~      \              a  \     a*   a    a
        Z    = Z    +  ) exp[-i G . R ]  )   P   dO    P
         nmx    nmx   /__            x  /__   ni   ii'  mi'

                       a                 ii'

    Note that this correction is an approximation that assumes the
    exponential varies slowly over the extent of the augmentation sphere.

    ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005)
    """

    if nbands is None:
        nbands = self.wfs.bd.nbands

    P_ani = self.wfs.kpt_u[u].P_ani
    P1_ani = self.wfs.kpt_u[u1].P_ani
    for a, P_ni in P_ani.items():
        P_ni = P_ani[a][:nbands]
        P1_ni = P1_ani[a][:nbands]
        dO_ii = self.wfs.setups[a].dO_ii
        e = np.exp(-2.j * np.pi * np.dot(G_c, self.spos_ac[a]))
        Z_nn += e * np.dot(np.dot(P_ni.conj(), dO_ii), P1_ni.T)
