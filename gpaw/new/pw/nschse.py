from gpaw.mpi import world


def nschse(ibzwfs, pathwfs=None, pathcomm=None):
    bzwfs = ibz2bz(ibzwfs)

            q_c = rskpt2.k_c - kpt1.k_c
            qd = KPointDescriptor([-q_c])
            pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
            ghat.set_positions(spos_ac)
            v_G = coulomb.get_potential(pd12)
            Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                                proj1[a],
                                Delta_iiL,
                                rskpt2.proj[a].conj())
                      for a, Delta_iiL in enumerate(paw.Delta_aiiL)]
            rho_nG = ghat.pd.empty(n2b - n2a, u1_nR.dtype)

            for n1, u1_R in enumerate(u1_nR):
                for u2_R, rho_G in zip(rskpt2.u_nR, rho_nG):
                    rho_G[:] = ghat.pd.fft(u1_R * u2_R.conj())

                ghat.add(rho_nG,
                         {a: Q_nnL[n1] for a, Q_nnL in enumerate(Q_annL)})

                for n2, rho_G in enumerate(rho_nG):
                    vrho_G = v_G * rho_G
                    e = ghat.pd.integrate(rho_G, vrho_G).real
                    e_nn[n1, n2] = e / kd.nbzkpts
            e_n -= e_nn.dot(rskpt2.f_n)

    for a, VV_ii in paw.VV_aii.items():
        P_ni = proj1all[a]
        vv_n = np.einsum('ni, ij, nj -> n',
                         P_ni.conj(), VV_ii, P_ni).real
        if paw.VC_aii[a] is not None:
            vc_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), paw.VC_aii[a], P_ni).real
        else:
            vc_n = 0.0
        E_n -= (2 * vv_n + vc_n)

    return E_n

