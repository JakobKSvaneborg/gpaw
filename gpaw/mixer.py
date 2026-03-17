# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
See Kresse, Phys. Rev. B 54, 11169 (1996)
"""

import numpy as np
from numpy.fft import fftn, ifftn

import gpaw.mpi as mpi
from gpaw.fd_operators import FDOperator
from gpaw.new import trace
from gpaw.utilities.blas import axpy
from gpaw.utilities.tools import construct_reciprocal

"""About mixing-related classes.

(FFT/Broyden)BaseMixer: These classes know how to mix one density
array and store history etc.  But they do not take care of complexity
like spin.

(SpinSum/etc.)MixerDriver: These combine one or more BaseMixers to
implement a full algorithm.  Think of them as stateless (immutable).
The user can give an object of these types as input, but they will generally
be constructed by a utility function so the interface is nice.

The density object always wraps the (X)MixerDriver with a
MixerWrapper.  The wrapper contains the common code for all mixers so
we don't have to implement it multiple times (estimate memory, etc.).

In the end, what the user provides is probably a dictionary anyway, and the
relevant objects are instantiated automatically."""

class BaseMixer:
#class BaseMixerOld:
    name = 'pulay'

    """Pulay density mixer."""
    def __init__(self, beta, nmaxold, weight):
        """Construct density-mixer object.

        Parameters:

        beta: float
            Mixing parameter between zero and one (one is most
            aggressive).
        nmaxold: int
            Maximum number of old densities.
        weight: float
            Weight parameter for special metric (for long wave-length
            changes).

        """

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        self.world = None

    def initialize_metric(self, gd):
        self.gd = gd

        if self.weight == 1:
            self.metric = None
        else:
            scale = 1 / np.sqrt(self.weight * 2)
            a = 0.125 * (self.weight + 7) * scale
            b = 0.0625 * (self.weight - 1) * scale
            c = 0.03125 * (self.weight - 1) * scale
            d = 0.015625 * (self.weight - 1) * scale
            self.metric = FDOperator([a,
                                      b, b, b, b, b, b,
                                      c, c, c, c, c, c, c, c, c, c, c, c,
                                      d, d, d, d, d, d, d, d],
                                     [(0, 0, 0),  # a
                                      (-1, 0, 0), (1, 0, 0),  # b
                                      (0, -1, 0), (0, 1, 0),
                                      (0, 0, -1), (0, 0, 1),
                                      (1, 1, 0), (1, 0, 1), (0, 1, 1),  # c
                                      (1, -1, 0), (1, 0, -1), (0, 1, -1),
                                      (-1, 1, 0), (-1, 0, 1), (0, -1, 1),
                                      (-1, -1, 0), (-1, 0, -1), (0, -1, -1),
                                      (1, 1, 1), (1, 1, -1), (1, -1, 1),  # d
                                      (-1, 1, 1), (1, -1, -1), (-1, -1, 1),
                                      (-1, 1, -1), (-1, -1, -1)],
                                     gd, float).apply

    def reset(self):
        """Reset Density-history.

        Called at initialization and after each move of the atoms.

        my_nuclei:   All nuclei in local domain.
        """

        # History for Pulay mixing of densities:
        self.nt_isG = []  # Pseudo-electron densities
        self.R_isG = []  # Residuals
        self.A_ii = np.zeros((0, 0))

        self.D_iasp = []
        self.dD_iasp = []

    def calculate_charge_sloshing(self, R_sG) -> float:
        return self.gd.integrate(np.fabs(R_sG)).sum()

    def mix_density(self, nt_sG, D_asp, g_ss=None, rhot=None):
        # nt_sG /= np.sum(nt_sG)
        nt_isG = self.nt_isG
        R_isG = self.R_isG
        D_iasp = self.D_iasp
        dD_iasp = self.dD_iasp
        spin = len(nt_sG)
        iold = len(self.nt_isG)
        dNt = np.inf
        if iold > 0:
            if iold > self.nmaxold:
                # Throw away too old stuff:
                del nt_isG[0]
                del R_isG[0]
                del D_iasp[0]
                del dD_iasp[0]
                # for D_p, D_ip, dD_ip in self.D_a:
                #     del D_ip[0]
                #     del dD_ip[0]
                iold = self.nmaxold

            # Calculate new residual (difference between input and
            # output density):
            R_sG = nt_sG - nt_isG[-1]
            dNt = self.calculate_charge_sloshing(R_sG)
            R_isG.append(R_sG)

            print('dnt: ', dNt)
            dD_iasp.append([])
            for D_sp, D_isp in zip(D_asp, D_iasp[-1]):
                dD_iasp[-1].append(D_sp - D_isp)

            if self.metric is None:
                mR_sG = R_sG
            else:
                mR_sG = np.empty_like(R_sG)
                for s in range(spin):
                    self.metric(R_sG[s], mR_sG[s])

            if g_ss is not None:
                mR_sG = np.tensordot(g_ss, mR_sG, axes=(1, 0))

            # Update matrix:
            A_ii = np.zeros((iold, iold))
            i2 = iold - 1

            for i1, R_1sG in enumerate(R_isG):
                a = self.dotprod(R_1sG, mR_sG, dD_iasp[i1], dD_iasp[-1], self.gd)
                A_ii[i1, i2] = a
                A_ii[i2, i1] = a
            A_ii[:i2, :i2] = self.A_ii[-i2:, -i2:]
            self.A_ii = A_ii

            try:
                B_ii = np.linalg.inv(A_ii)
                alpha_i = B_ii.sum(1)
                alpha_i /= alpha_i.sum()
            except (ZeroDivisionError, np.linalg.LinAlgError):
                alpha_i = np.zeros(iold)
                alpha_i[-1] = 1.0

            if self.world:
                self.world.broadcast(alpha_i, 0)

            # Calculate new input density:
            nt_sG[:] = 0.0
            # for D_p, D_ip, dD_ip in self.D_a:
            for D in D_asp:
                D[:] = 0.0
            beta = self.beta
            for i, alpha in enumerate(alpha_i):
                axpy(alpha, nt_isG[i], nt_sG)
                axpy(alpha * beta, R_isG[i], nt_sG)

                for D_sp, D_isp, dD_isp in zip(D_asp, D_iasp[i],
                                               dD_iasp[i]):
                    axpy(alpha, D_isp, D_sp)
                    axpy(alpha * beta, dD_isp, D_sp)

        # Store new input density (and new atomic density matrices):
        nt_isG.append(nt_sG.copy())
        D_iasp.append([])
        for D_sp in D_asp:
            D_iasp[-1].append(D_sp.copy())
        return dNt

    def dotprod(self, R1_isG, R2_isG, dD1_iasp, dD2_iasp, gd, mode='scalar'):
        comm = gd.comm

        if mode == 'scalar':
            R1_isG = [R1_isG, ]
            dD1_iasp = [dD1_iasp, ]
        if mode == 'scalar':
            R2_isG = [R2_isG, ]
            dD2_iasp = [dD2_iasp, ]

        if mode == 'gemm':
            shape1 = np.array(R1_isG).shape
            shape2 = np.array(R2_isG).shape
            prod = np.reshape(R1_isG, (shape1[0], -1)).conj() \
                 @ np.reshape(R2_isG, (shape2[0], -1)).T
        elif mode == 'vecdot' or mode == 'scalar':
            assert len(R1_isG) == len(R2_isG)
            prod = np.vecdot(
                np.reshape(R1_isG, (len(R1_isG), -1)),
                np.reshape(R2_isG, (len(R2_isG), -1))
            )
        prod *= gd.dv
        comm.sum(prod)
        assert (prod.imag < 1e-10).all()
        prod = prod.real
        return prod[0] if mode == 'scalar' else prod

    def estimate_memory(self, mem, gd):
        gridbytes = gd.bytecount()
        mem.subnode('nt_iG, R_iG', 2 * self.nmaxold * gridbytes)

    def __repr__(self):
        classname = self.__class__.__name__
        template = '%s(beta=%f, nmaxold=%d, weight=%f)'
        string = template % (classname, self.beta, self.nmaxold, self.weight)
        return string

class MSR1Mixer(BaseMixer):
    name = 'MSR1'
    soft_bad_lim = 1.5
    hard_bad_lim = 2.0
    panic_threshold = 50.0
    has_panicked = True

    dNt_i = []
    def reset(self):
        super().reset()
        self.dNt_i = []
        self.last_dNt = np.inf

    def mix_density(self, nt_sG, D_asp, g_ss=None):
        nt_isG = self.nt_isG
        R_isG = self.R_isG
        D_iasp = self.D_iasp
        dD_iasp = self.dD_iasp
        iold = len(self.nt_isG)
        dNt = np.inf
        if iold > 0:
            # Calculate new residual (difference between input and
            # output density):
            self.dNt_i = [dNtt * 1.2 for dNtt in self.dNt_i]
            R_sG = nt_sG - nt_isG[-1]
            dNt = self.calculate_charge_sloshing(R_sG)
            self.dNt_i.append(dNt)
            R_isG.append(R_sG)
            dD_iasp.append([])
            for D_sp, D_isp in zip(D_asp, D_iasp[-1]):
                dD_iasp[-1].append(D_sp - D_isp)

            if dNt > self.last_dNt * self.panic_threshold or (iold > 1 and not self.has_panicked):
                if self.world.rank == 0:
                    print(f'XXXX: PANIC, {dNt} < {self.last_dNt * self.panic_threshold}')
                self.has_panicked = True
                self.panic_threshold *= 1.2
                last_dens = self.nt_isG[-2]
                last_paw = self.D_iasp[-2]
                last_res = self.R_isG[-2]
                last_res_paw = self.dD_iasp[-2]
                self.reset()
                iold = 1
                self.dNt_i = [self.last_dNt, ]
                self.nt_isG = [last_dens, ]
                nt_isG = self.nt_isG
                self.R_isG = [last_res, ]
                R_isG = self.R_isG
                self.D_iasp = [last_paw, ]
                D_iasp = self.D_iasp
                self.dD_iasp = [last_res_paw, ]
                dD_iasp = self.dD_iasp
                R_sG = last_res
                dNt = self.last_dNt


            while (iold > self.nmaxold and dNt <= self.last_dNt * self.soft_bad_lim) \
                    or (iold > self.nmaxold):
                # Throw away too old stuff:
                # to_del = np.argmax(np.array(self.dNt_i)[:-2])
                to_del = 0
                assert len(self.dNt_i) == len(nt_isG)
                del nt_isG[to_del]
                del R_isG[to_del]
                del D_iasp[to_del]
                del dD_iasp[to_del]
                del self.dNt_i[to_del]
                if self.world.rank == 0:
                    print('Deleting index ', to_del)
                # for D_p, D_ip, dD_ip in self.D_a:
                #     del D_ip[0]
                #     del dD_ip[0]
                iold = len(nt_isG)

        if self.world.rank == 0:
            print('hist: ', iold)
        if iold > 1:
            backtracked = False
            last_step = -2
            del_oldest = False
            if dNt > self.last_dNt * self.soft_bad_lim:
                reduction = dNt / self.last_dNt
                if self.world.rank == 0:
                    print(f'XXX: Backtracing due to reduction {reduction}')
                del_oldest = True if reduction > self.hard_bad_lim else False
                insert_pos = 0 if del_oldest else -2
                last_step = -1
                tmp = nt_isG.pop()
                nt_isG.insert(insert_pos, tmp)
                tmp = R_isG.pop()
                R_isG.insert(insert_pos, tmp)
                tmp = D_iasp.pop()
                D_iasp.insert(insert_pos, tmp)
                tmp = dD_iasp.pop()
                dD_iasp.insert(insert_pos, tmp)
                dNt = self.dNt_i.pop()
                self.dNt_i.insert(insert_pos, dNt)
                dNt = self.last_dNt * 1.1  # Avoid infinite loop

                R_sG = R_isG[-1]
                backtracked = True

            # 1st order norm
            ntnorm = self.calculate_charge_sloshing(nt_isG[-1])
            dNt_normed = dNt / ntnorm
            if self.world.rank == 0:
            #     print(f"Normed dNt: {np.log(dNt_normed)}")
                print(f"Max broyden: {1e-2 / dNt_normed}")

            dampen = 1.0  # Dampen the greeds
            punishment_factor = 0.85 if del_oldest else 1  # How much to reduce greed when backtracing
            trust_scalar = 1.1 # Scaling factor for the trust radius.
            # if self.trust_radius is not None:
            #     self.trust_radius *= punishment_factor
            abs_gb_lim = 1000 # Maximum value of good Broyden.
            max_gb_fact = 0.3 * np.clip(1e-2 / dNt_normed, 0.05, 1) # Scaling factor for maximum good Broyden.
            post_gb_fact = 0.9 if del_oldest else (0.9 if backtracked else 0.9)  # Scaling factor for the final amount of good Broyden
            weight = 5e-5  # Weight for regularization, 2e-4 works well. Strongly depends on the amount of good Broyden.
            B0_boost = 2e-1  # Favor the predicted greed towards 1
            B0_lims = [0.2, 1.1]   # Limits for predicted greed
            A0_lims = [0.02, 0.4]   # Limits for unpredicted greed
            rate_ratio = [0.7, 1.3 if not backtracked else punishment_factor]  # Rate ratio for clipping
            renormalize = True  # Renormalize t_isG
            initial_B0 = 1.0

            # 2nd order norm
            # ntnorm_i = np.vecdot(np.array(nt_isG).reshape(iold, -1),
            #     np.array(nt_isG).reshape(iold, -1))
            # self.gd.comm.sum(ntnorm_i)
            # ntnorm_i = np.expand_dims(1 / np.sqrt(ntnorm_i), axis=tuple(np.arange(1, np.array(nt_isG).ndim)))

            ### 2023 Paper, Eq: 8 + 9:
            s_isG = (nt_isG[:-1] - nt_isG[-1])
            y_isG = -(R_isG[:-1] - R_isG[-1])

            sD_iasp = []
            yD_iasp = []
            for i1 in range(len(D_iasp) - 1):
                sD_asp = []
                yD_asp = []
                for a1 in range(len(D_iasp[i1])):
                    sD_asp.append((D_iasp[i1][a1] - D_iasp[-1][a1]))
                    yD_asp.append(-(dD_iasp[i1][a1] - dD_iasp[-1][a1]))
                sD_iasp.append(sD_asp)
                yD_iasp.append(yD_asp)

            metric = self.metric

            ###
            ts_isG = s_isG.copy()
            ty_isG = y_isG.copy()
            for ty_sG, ts_sG in zip(ty_isG, ts_isG):
                for ty_G, ts_G in zip(ty_sG, ts_sG):
                    if metric is not None:
                        self.metric(ty_G.copy(), ty_G)
                        # self.metric(ts_G.copy(), ts_G)
                if g_ss is not None:
                    ty_sG[:] = np.tensordot(g_ss, ty_sG, axes=(1, 0))
                    ts_sG[:] = np.tensordot(g_ss, ts_sG, axes=(1, 0))

            tsD_iasp = []
            tyD_iasp = []
            for sD_asp, yD_asp in zip(sD_iasp, yD_iasp):
                tyD_asp = []
                tsD_asp = []
                for sD_sp, yD_sp in zip(sD_asp, yD_asp):
                    if g_ss is not None:
                        tyD_sp = np.tensordot(g_ss, yD_sp, axes=(1, 0))
                        tsD_sp = np.tensordot(g_ss, sD_sp, axes=(1, 0))
                    else:
                        tyD_sp = yD_sp.copy()
                        tsD_sp = sD_sp.copy()
                    tyD_asp.append(tyD_sp)
                    tsD_asp.append(tsD_sp)
                tsD_iasp.append(tsD_asp)
                tyD_iasp.append(tyD_asp)

            ### Scale the problem!
            y_norm = self.dotprod(ty_isG, y_isG, tyD_iasp, yD_iasp, self.gd, mode='vecdot')
            s_norm = self.dotprod(ts_isG, y_isG, tsD_iasp, yD_iasp, self.gd, mode='vecdot')
            t_norm = 1 / np.sqrt(y_norm) #  + 0.5 * max_gb * s_norm)
            t_norm_expanded = np.expand_dims(t_norm, axis=tuple(np.arange(1, y_isG.ndim)))
            # XXX:
            # t_norm = np.sum(ty_isG * y_isG, axis=tuple(np.arange(2, y_isG.ndim)))
            # self.gd.comm.sum(t_norm)
            # t_norm = 1 / np.sqrt(t_norm)
            # t_norm_expanded = np.expand_dims(t_norm, axis=tuple(np.arange(2, y_isG.ndim)))
            # XXX
            y_isG *= t_norm_expanded
            ty_isG *= t_norm_expanded
            s_isG *= t_norm_expanded
            ts_isG *= t_norm_expanded
            for i, (sD_asp, yD_asp, tsD_asp, tyD_asp) in enumerate(zip(sD_iasp, yD_iasp, tsD_iasp, tyD_iasp)):
                for a, (sD_sp, yD_sp, tsD_sp, tyD_sp) in enumerate(zip(sD_asp, yD_asp, tsD_asp, tyD_asp)):
                    sD_sp *= t_norm[i] # [:, None]
                    yD_sp *= t_norm[i] # [:, None]
                    tsD_sp *= t_norm[i] # [:, None]
                    tyD_sp *= t_norm[i] # [:, None]
            y_norm = self.dotprod(ty_isG, y_isG, tyD_iasp, yD_iasp, self.gd, mode='vecdot')
            s_norm = self.dotprod(ts_isG, y_isG, tsD_iasp, yD_iasp, self.gd, mode='vecdot')

            ### 2023 paper eq 22 - Limit good_broydenness
            # YY_LIM = y_isG.reshape((iold - 1, -1)) @ ty_isG.reshape((iold - 1, -1)).T
            YY_LIM = self.dotprod(ty_isG, y_isG, tyD_iasp, yD_iasp, self.gd, mode='gemm')
            YY_LIM = np.linalg.norm(YY_LIM, ord='fro')
            YS_LIM = self.dotprod(ts_isG, y_isG, tsD_iasp, yD_iasp, self.gd, mode='gemm')
            YS_LIM = np.linalg.norm(YS_LIM, ord='fro')
            max_gb = min(max(YY_LIM / YS_LIM * max_gb_fact, 1), abs_gb_lim)
            good_broydenness = 0.5 * max_gb

            # A_ii0 = self.dotprod(ty_isG, y_isG, tyD_iasp, yD_iasp, self.gd, mode='gemm')
            # norm_vec = 1 / np.sqrt(y_norm)
            # A_ii0 *= norm_vec[None, :]
            # A_ii0 *= norm_vec[:, None]
            # eigval0 = np.min(np.linalg.eigvals(A_ii0))

            # Choose max good_broydenness s.t. A_ii is positive definite
            # for good_broydenness in good_broydenness_range:
            # binary search 2**(-8) accuracy:
            for iter in range(2, 9):
                t_isG = ty_isG + good_broydenness * ts_isG
                tD_iasp = []
                for i in range(iold - 1):
                    tD_asp = []
                    for a in range(len(D_iasp[0])):
                        tD_asp.append(tyD_iasp[i][a] + good_broydenness * tsD_iasp[i][a])
                    tD_iasp.append(tD_asp)

                norm_vec = 1 / np.sqrt(np.abs(y_norm + s_norm * good_broydenness))

                A_ii = self.dotprod(t_isG, y_isG, tD_iasp, yD_iasp, self.gd, mode='gemm')
                A_ii *= norm_vec[None, :]
                A_ii *= norm_vec[:, None]
                try:
                    eigs = np.linalg.eigvals(A_ii)
                    min_real = np.min(eigs.real)
                    max_imag = np.max(np.abs(eigs.imag))
                except np.linalg.LinAlgError:
                    good_broydenness -= 2**(-iter) * max_gb
                    continue
                if min_real > max(1e-9, max_imag):
                    good_broydenness += 2**(-iter) * max_gb
                else:
                    good_broydenness -= 2**(-iter) * max_gb
            good_broydenness -= 2**(-iter-1) * max_gb
            good_broydenness *= post_gb_fact

            # good_broydenness = self.last_gb * min(gb_factor, 2)
            # Do not good broyden when density is crap
            # crabiness_mult = -np.log(2 * dNt * ntnorm_i.ravel()[-1]) / 3
            # good_broydenness *= np.clip(crabiness_mult, 0.1, 1)
            if self.world.rank == 0:
                print('good_broydenness: ', good_broydenness)

            ### Re-Scale the problem!
            if renormalize:
                t_norm = self.dotprod(ty_isG, y_isG, yD_iasp, yD_iasp, self.gd, mode='vecdot')
                t_norm += good_broydenness \
                    * self.dotprod(ts_isG, y_isG, sD_iasp, yD_iasp, self.gd, mode='vecdot')
                t_norm = 1 / np.sqrt(t_norm)
                t_norm_expanded = np.expand_dims(t_norm, axis=tuple(np.arange(1, y_isG.ndim)))
                y_isG *= t_norm_expanded
                ty_isG *= t_norm_expanded
                s_isG *= t_norm_expanded
                ts_isG *= t_norm_expanded
                for i, (sD_asp, yD_asp, tsD_asp, tyD_asp) in enumerate(zip(sD_iasp, yD_iasp, tsD_iasp, tyD_iasp)):
                    for sD_sp, yD_sp, tsD_sp, tyD_sp in zip(sD_asp, yD_asp, tsD_asp, tyD_asp):
                        sD_sp *= t_norm[i]
                        yD_sp *= t_norm[i]
                        tsD_sp *= t_norm[i]
                        tyD_sp *= t_norm[i]

            t_isG = ty_isG + good_broydenness * ts_isG
            tD_iasp = []
            for i in range(iold - 1):
                tD_asp = []
                for a in range(len(D_iasp[0])):
                    tD_asp.append(tyD_iasp[i][a] + good_broydenness * tsD_iasp[i][a])
                tD_iasp.append(tD_asp)

            A_ii = self.dotprod(t_isG, y_isG, tD_iasp, yD_iasp, self.gd, mode='gemm')
            B_ii = self.dotprod(t_isG, s_isG, tD_iasp, sD_iasp, self.gd, mode='gemm')
            # B_ii = self.dotprod(ty_isG, s_isG, tyD_iasp, sD_iasp, self.gd, mode='gemm')

            A_diag = np.mean(np.diag(A_ii))
            B_diag = np.mean(np.diag(B_ii))

            ### SVD Regularization:
            S, V, D = np.linalg.svd(B_ii)
            # V = V / (V**2 + (1e-10 * np.max(V))**2)
            V = V / (V**2 + B_diag**2 * weight**2)
            B_ii = D.T @ np.diag(V) @ S.T
            # B_ii = np.linalg.inv(B_ii)

            S, V, D = np.linalg.svd(A_ii)
            # A_i = V / (V**2 + (weight * np.max(V))**2)
            A_i = V / (V**2 + A_diag**2 * weight**2)
            A_ii = D.T @ np.diag(A_i) @ S.T

            # 2023 paper eq 14 alpha_i = Inv(Y_n^T @ W) @ W^T @ Res_n part
            # A_ii should not be transposed... Or should be... Depending on
            # what paper your read, transposed works best, but most papers
            # say not to, so... rip

            alpha_i = self.dotprod(t_isG, [R_sG, ], tD_iasp, [dD_iasp[-1], ], self.gd, mode='gemm')[:, 0]
            alpha_i = (A_ii @ alpha_i).real

            if self.world:
                self.world.broadcast(alpha_i, 0)

            ######### Stuff for predicting mixing coefficients:
            uRnoD_asp = []
            for uD_sp, R_sp in zip(self.uD_asp, dD_iasp[last_step]):
                uRnoD_asp.append(R_sp - uD_sp)

            tuk_sG = self.uk_sG.copy()
            tuRnoD_sG = self.R_isG[last_step] - self.uk_sG
            if False and metric is not None:
                self.metric(tuk_sG, tuk_sG)
                self.metric(tuRnoD_sG, tuRnoD_sG)

            A1 = self.dotprod(tuk_sG, self.uk_sG, self.uD_asp, self.uD_asp,
               self.gd, mode='scalar')

            B1 = self.dotprod(tuRnoD_sG,
                self.R_isG[last_step] - self.uk_sG,
                uRnoD_asp, uRnoD_asp, self.gd, mode='scalar')

            # From Eq 18 from mixing for dumies:
            A2_i = self.dotprod(t_isG, [self.uk_sG, ], tD_iasp, [self.uD_asp, ], self.gd, mode='gemm')[:, 0]
            # A2_i = self.dotprod(ty_isG, [self.uk_sG, ], tyD_iasp, [self.uD_asp, ], self.gd, mode='gemm')[:, 0]
            A3_i = self.dotprod([tuk_sG, ], y_isG, [self.uD_asp, ], yD_iasp, self.gd, mode='gemm')[0, :]

            B2_i = self.dotprod(t_isG, [self.pk_sG, ], tD_iasp, [self.pD_asp, ], self.gd, mode='gemm')[:, 0]
            # B2_i = self.dotprod(ty_isG, [self.pk_sG, ], tyD_iasp, [self.pD_asp, ], self.gd, mode='gemm')[:, 0]
            B3_i = self.dotprod([tuRnoD_sG, ], y_isG, [uRnoD_asp, ], yD_iasp, self.gd, mode='gemm')[0, :]


            A2 = A3_i @ B_ii @ A2_i * dampen
            B2 = B3_i @ B_ii @ B2_i

            if self.world.rank == 0:
                print('ratio: ', np.abs(A1 / A2))

            A0_target = np.clip(
                np.arctan(np.pi * np.abs(A1 / A2) / A0_lims[1] * 0.5) / np.pi * 2 * A0_lims[1],
                # np.abs(A1 / A2),
                *A0_lims
                )
            if self.A0 is not None:
                if B2 == 0 and B1 == 0:
                    B2 = 1; B1 = 1
                B0_ratio = (
                    B0_boost + self.B0 + np.clip(np.abs(B1 / B2), *B0_lims)
                    ) / ((2 + B0_boost) * self.B0)
                self.B0 *= np.clip(B0_ratio, 3/5, 5/3)
                self.B0 = np.clip(self.B0, *B0_lims)
                A0_ratio_GEOM = np.sqrt(A0_target * self.A0) / self.A0
                A0_ratio_ALG = (A0_target + self.A0) / (2 * self.A0)
                self.last_A0_rate = np.clip(A0_ratio_GEOM, *rate_ratio)
                self.A0 *= self.last_A0_rate
            else:
                self.B0 = initial_B0
                self.A0 = np.clip(A0_target, *A0_lims)
                self.last_A0_rate = np.clip((self.A0 / self.beta)**0.5, *rate_ratio)
                self.last_B0_rate = 1.0

            A0 = self.A0
            B0 = self.B0
            if self.world.rank == 0:
                print(f"rank: {self.world.rank}, A0: {A0}, B0: {B0}")

            trust_step = (A0 * self.uk_sG + B0 * self.pk_sG)
            dstep_asp = []
            for uD_sp, pD_sp in zip(self.uD_asp, self.pD_asp):
                dstep_asp.append(A0 * uD_sp + B0 * pD_sp)
            trust_radius = self.dotprod(trust_step, trust_step, dstep_asp,
               dstep_asp, self.gd, mode='scalar')
            trust_radius = trust_scalar * trust_radius**0.5

            if self.trust_radius is not None:
                trust_radius_factor = (self.trust_radius * trust_radius) ** 0.5 / self.trust_radius
                self.trust_radius *= np.clip(trust_radius_factor, *rate_ratio)
            else:
                self.trust_radius = trust_radius * 0.7

            self.uk_sG = np.zeros_like(nt_sG)
            self.pk_sG = np.zeros_like(nt_sG)

            for pD_sp in self.pD_asp:
                pD_sp[:] = 0

            for i1, alpha in enumerate(alpha_i):
                self.uk_sG -= y_isG[i1] * alpha
                self.pk_sG += s_isG[i1] * alpha

                for a1, sD_sp in enumerate(sD_iasp[i1]):
                    self.pD_asp[a1] += alpha * sD_sp

            self.uk_sG += R_sG
            step_sG = A0 * self.uk_sG + B0 * self.pk_sG
            step_size = B0 * self.dotprod(self.pk_sG, self.pk_sG,
                self.pD_asp, self.pD_asp, self.gd, mode='scalar')**0.5

            beta_i = alpha_i.copy()

            # if self.world.rank == 0:
            #     print(f"Step size: {step_size}, trust_radius: {self.trust_radius}")

            if step_size > self.trust_radius * 1.02:
                # Time to mix the mixing...
                if self.world.rank == 0:
                    print(f'XXXX {step_size} > {self.trust_radius}')
                ### Perform lsq squares with lagrange multiplier
                # B^T R:
                BR_i = self.dotprod(t_isG, [R_sG, ], tD_iasp, [dD_iasp[-1]], self.gd, mode='gemm')[:, 0]

                # f^T f
                s_ii = self.dotprod(s_isG, s_isG, sD_iasp, sD_iasp, self.gd, mode='gemm') * B0**2

                A_ii = np.linalg.inv(A_ii)
                # Optimize (ridge regression):
                def err_fct(lamb):
                    beta_i = np.linalg.solve(
                        A_ii + lamb * np.eye(iold - 1), BR_i
                    )
                    # tA_i = V / (V**2 + A_diag**2 * lamb**2)
                    # beta_i = (D.T @ np.diag(tA_i) @ S.T) @ BR_i
                    return (beta_i @ s_ii @ beta_i) - self.trust_radius**2

                from scipy.optimize import root_scalar
                try:
                    lamb = root_scalar(err_fct, bracket=[0, 100 * np.max(V)])
                    # lamb = root_scalar(err_fct, bracket=[weight, 10])
                    root = lamb.root
                except ValueError as e:
                    print('XXX: Failed to match trust region!!')
                    root = 100 * np.max(V)
                    # root = 10
                beta_i = np.linalg.solve(
                    A_ii + root * np.eye(iold - 1), BR_i
                )
                # tA_i = V / (V**2 + root**2 * A_diag**2)
                # beta_i = (D.T @ np.diag(tA_i) @ S.T) @ BR_i
                if self.world:
                    self.world.broadcast(beta_i, 0)

                self.uk_sG[:] = 0
                self.pk_sG[:] = 0
                for pD_sp in self.pD_asp:
                    pD_sp[:] = 0
                alpha_i[:] = beta_i

                for i1, (alpha, beta) in enumerate(zip(alpha_i, beta_i)):
                    self.uk_sG -= y_isG[i1] * alpha
                    self.pk_sG += s_isG[i1] * beta

                    for a1, sD_sp in enumerate(sD_iasp[i1]):
                        self.pD_asp[a1] += beta * sD_sp

                # new_step_size = B0**2 * self.dotprod(self.pk_sG, self.pk_sG,
                #     self.pD_asp, self.pD_asp, self.gd, mode='scalar')
                # new_step_size = new_step_size**0.5
                new_step_size = self.trust_radius
                scale_factor = (new_step_size / step_size)
                A0 *= np.clip(scale_factor, 0, 1)
                A0 = max(A0, min(self.A0, A0_lims[0]))
                self.A0 = A0  # (self.A0 + A0) / 2
                self.uk_sG += R_sG
                step_sG = A0 * self.uk_sG + B0 * self.pk_sG

            nt_sG[:] = nt_isG[-1] + step_sG

            self.uD_asp = []
            self.pD_asp = []
            for a1, D_sp in enumerate(D_asp):
                D_sp[:] = D_iasp[-1][a1] + A0 * dD_iasp[-1][a1]
                self.uD_asp.append(dD_iasp[-1][a1].copy())
                self.pD_asp.append(np.zeros_like(D_sp))

            for i1, (alpha, beta) in enumerate(zip(alpha_i, beta_i)):
                for a1, D_sp in enumerate(D_asp):
                    D_sp -= A0 * alpha * yD_iasp[i1][a1]
                    D_sp += B0 * beta * sD_iasp[i1][a1]
                    self.uD_asp[a1] -= alpha * yD_iasp[i1][a1]
                    self.pD_asp[a1] += beta * sD_iasp[i1][a1]

            # Sync the density, because apparantly they cant agree...
            if self.world:
                nt_sR = self.gd.collect(nt_sG, broadcast=True)
                self.world.broadcast(nt_sR, 0)
                nt_sG[:] = self.gd.distribute(nt_sR)

            if del_oldest:
                del nt_isG[0]
                del R_isG[0]
                del dD_iasp[0]
                del D_iasp[0]
                del self.dNt_i[0]

        elif iold > 0:
            # Pratt step
            self.i_update = 0
            self.last_gb = 5
            self.trust_radius = None
            self.A0 = None
            A0 = self.beta
            self.uk_sG = R_sG
            self.pk_sG = np.zeros_like(self.uk_sG)
            nt_sG[:] = nt_isG[-1] + A0 * self.uk_sG
            self.uD_asp = []
            self.pD_asp = []
            for a1, D_sp in enumerate(D_asp):
                D_sp[:] = D_iasp[-1][a1] + A0 * dD_iasp[-1][a1]
                self.uD_asp.append(dD_iasp[-1][a1].copy())
                self.pD_asp.append(np.zeros_like(D_sp))

        # Store new input density (and new atomic density matrices):
        nt_isG.append(nt_sG.copy())
        D_iasp.append([])
        for D_sp in D_asp:
            D_iasp[-1].append(D_sp.copy())
        self.last_dNt = dNt
        return dNt


class ExperimentalDotProd:
    def __init__(self, setups, atomdist):
        self.setups = setups
        self.atomdist = atomdist

    def __call__(self, R1_isG, R2_isG, dD1_iasp, dD2_iasp, gd, mode='scalar'):
        from gpaw.utilities import pack_density, unpack_density, unpack_hermitian
        setups = self.setups
        comm = gd.comm

        if mode == 'scalar':
            R1_isG = [R1_isG, ]
            dD1_iasp = [dD1_iasp, ]
        if mode == 'scalar':
            R2_isG = [R2_isG, ]
            dD2_iasp = [dD2_iasp, ]

        if mode == 'gemm':
            shape1 = np.array(R1_isG).shape
            shape2 = np.array(R2_isG).shape
            prod = np.reshape(R1_isG, (shape1[0], -1)).conj() \
                 @ np.reshape(R2_isG, (shape2[0], -1)).T
        elif mode == 'vecdot' or mode == 'scalar':
            assert len(R1_isG) == len(R2_isG)
            prod = np.vecdot(
                np.reshape(R1_isG, (len(R1_isG), -1)),
                np.reshape(R2_isG, (len(R2_isG), -1))
            )
        prod *= gd.dv
        assert self.atomdist.comm.rank == comm.rank
        my_atoms_inds = np.where(self.atomdist.rank_a == comm.rank)[0]
        for a, a_s in enumerate(my_atoms_inds):
            setup = setups[a_s]
            ni = setup.ni
            I4_pp = setup.four_phi_integrals()
            I4_pp = unpack_hermitian(I4_pp).reshape(-1, ni**2).T.copy()
            I4_pp = unpack_hermitian(I4_pp).reshape(ni**2, ni**2)

            if mode == 'gemm':
                for i1, dD1_asp in enumerate(dD1_iasp):
                    dD1_sp = dD1_asp[a].conj()  # pack_density(dD1_asp[a].reshape(-1, ni, ni))
                    for i2, dD2_asp in enumerate(dD2_iasp):
                        dD2_sp = dD2_asp[a]  # pack_density(dD2_asp[a].reshape(-1, ni, ni))
                        for dD1_p, dD2_p in zip(dD1_sp, dD2_sp):
                            prod[i1, i2] += dD1_p @ I4_pp @ dD2_p
            elif mode == 'vecdot' or mode == 'scalar':
                for i, (dD1_asp, dD2_asp) in enumerate(zip(dD1_iasp, dD2_iasp)):
                    dD1_sp = dD1_asp[a].conj()  # pack_density(dD1_asp[a].reshape(-1, ni, ni))
                    dD2_sp = dD2_asp[a]  # pack_density(dD2_asp[a].reshape(-1, ni, ni))
                    for dD1_p, dD2_p in zip(dD1_sp, dD2_sp):
                        prod[i] += dD1_p @ I4_pp @ dD2_p
        comm.sum(prod)
        assert (prod.imag < 1e-10).all()
        prod = prod.real
        if mode == 'scalar':
            assert prod.size == 1
        return prod[0] if mode == 'scalar' else prod


class ReciprocalMetric:
    def __init__(self, weight, k2_Q, gd):
        k2_min = np.min(k2_Q)
        self.q1 = (weight - 1) * k2_min
        self.k2_Q = gd.distribute(k2_Q)

    def __call__(self, R_Q, mR_Q):
        mR_Q[:] = R_Q * (1.0 + self.q1 / self.k2_Q)


class FFTBaseMixer(MSR1Mixer):
    name = 'fft'

    """Mix the density in Fourier space"""
    def __init__(self, beta, nmaxold, weight):
        super().__init__(beta, nmaxold, weight)
        self.gd1 = None

    def initialize_metric(self, gd):
        self.gd = gd

        # if gd.comm.rank == 0:
        self.gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        k2_Q, _ = construct_reciprocal(self.gd1)
        self.metric = ReciprocalMetric(self.weight, k2_Q, self.gd)

        #     self.metric = ReciprocalMetric(self.weight, k2_Q)
        # else:
        #     self.metric = lambda R_Q, mR_Q: None

    def calculate_charge_sloshing(self, R_sQ):
        assert R_sQ.ndim == 4  # and len(R_sQ) == 1
        cs = 0.0
        for R_Q in R_sQ:
            R_X = self.gd.collect(R_Q)
            if self.gd.comm.rank == 0:
                cs += self.gd1.integrate(np.abs(ifftn(R_X, norm='ortho')).real)
            # cs = sum(self.gd1.integrate(np.abs(ifftn(R_Q, norm='ortho')).real)
            #         for R_Q in R_sQ)

        return self.gd.comm.sum_scalar(cs)

    def mix_density(self, nt_sR, D_asp, g_ss=None, rhot=None):
        # Transform real-space density to Fourier space
        nt1_sR = [self.gd.collect(nt_R) for nt_R in nt_sR]
        if self.gd.comm.rank == 0:
            nt1_sG = np.ascontiguousarray([fftn(nt_R, norm='ortho') for nt_R in nt1_sR])
        else:
            nt1_sG = np.empty((len(nt_sR), 0, 0, 0), dtype=complex)
        nt_sG = np.array([self.gd.distribute(nt1_G) for nt1_G in nt1_sG])

        dNt = super().mix_density(nt_sG, D_asp)

        nt1_sG = [self.gd.collect(nt_G) for nt_G in nt_sG]
        # Return density in real space
        for nt_G, nt_R in zip(nt1_sG, nt_sR):
            if self.gd.comm.rank == 0:
                nt1_R = ifftn(nt_G, norm='ortho').real
            else:
                nt1_R = None
            self.gd.distribute(nt1_R, nt_R)

        return dNt


class BroydenBaseMixer:
    name = 'broyden'

    def __init__(self, beta, nmaxold, weight):
        self.verbose = False
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = 1.0  # XXX discards argument

    def initialize_metric(self, gd):
        self.gd = gd

    def reset(self):
        self.step = 0
        # self.d_nt_G = []
        # self.d_D_ap = []

        self.R_iG = []
        self.dD_iap = []

        self.nt_iG = []
        self.D_iap = []
        self.c_G = []
        self.v_G = []
        self.u_G = []
        self.u_D = []

    def mix_density(self, nt_sG, D_asp, g_ss=None):
        if g_ss is not None:
            raise NotImplementedError()

        nt_G = nt_sG[0]
        D_ap = [D_sp[0] for D_sp in D_asp]
        dNt = np.inf
        if self.step > 2:
            del self.R_iG[0]
            for d_Dp in self.dD_iap:
                del d_Dp[0]
        if self.step > 0:
            self.R_iG.append(nt_G - self.nt_iG[-1])
            for d_Dp, D_p, D_ip in zip(self.dD_iap, D_ap, self.D_iap):
                d_Dp.append(D_p - D_ip[-1])
            fmin_G = self.gd.integrate(self.R_iG[-1] * self.R_iG[-1])
            dNt = self.gd.integrate(np.fabs(self.R_iG[-1]))
            if self.verbose:
                print('Mixer: broyden: fmin_G = %f fmin_D = %f' % fmin_G)
        if self.step == 0:
            self.eta_G = np.empty(nt_G.shape)
            self.eta_D = []
            for D_p in D_ap:
                self.eta_D.append(0)
                self.u_D.append([])
                self.D_iap.append([])
                self.dD_iap.append([])
        else:
            if self.step >= 2:
                del self.c_G[:]
                if len(self.v_G) >= self.nmaxold:
                    del self.u_G[0]
                    del self.v_G[0]
                    for u_D in self.u_D:
                        del u_D[0]
                temp_nt_G = self.R_iG[1] - self.R_iG[0]
                self.v_G.append(temp_nt_G / self.gd.integrate(temp_nt_G *
                                                              temp_nt_G))
                if len(self.v_G) < self.nmaxold:
                    nstep = self.step - 1
                else:
                    nstep = self.nmaxold
                for i in range(nstep):
                    self.c_G.append(self.gd.integrate(self.v_G[i] *
                                                      self.R_iG[1]))
                self.u_G.append(self.beta * temp_nt_G + self.nt_iG[1] -
                                self.nt_iG[0])
                for d_Dp, u_D, D_ip in zip(self.dD_iap, self.u_D, self.D_iap):
                    temp_D_ap = d_Dp[1] - d_Dp[0]
                    u_D.append(self.beta * temp_D_ap + D_ip[1] - D_ip[0])
                usize = len(self.u_G)
                for i in range(usize - 1):
                    a_G = self.gd.integrate(self.v_G[i] * temp_nt_G)
                    axpy(-a_G, self.u_G[i], self.u_G[usize - 1])
                    for u_D in self.u_D:
                        axpy(-a_G, u_D[i], u_D[usize - 1])
            self.eta_G = self.beta * self.R_iG[-1]
            for i, d_Dp in enumerate(self.dD_iap):
                self.eta_D[i] = self.beta * d_Dp[-1]
            usize = len(self.u_G)
            for i in range(usize):
                axpy(-self.c_G[i], self.u_G[i], self.eta_G)
                for eta_D, u_D in zip(self.eta_D, self.u_D):
                    axpy(-self.c_G[i], u_D[i], eta_D)
            axpy(-1.0, self.R_iG[-1], nt_G)
            axpy(1.0, self.eta_G, nt_G)
            for D_p, d_Dp, eta_D in zip(D_ap, self.dD_iap, self.eta_D):
                axpy(-1.0, d_Dp[-1], D_p)
                axpy(1.0, eta_D, D_p)
            if self.step >= 2:
                del self.nt_iG[0]
                for D_ip in self.D_iap:
                    del D_ip[0]
        self.nt_iG.append(np.copy(nt_G))
        for D_ip, D_p in zip(self.D_iap, D_ap):
            D_ip.append(np.copy(D_p))
        self.step += 1
        return dNt


class DummyMixer:
    """Dummy mixer for TDDFT, i.e., it does not mix."""
    name = 'dummy'
    beta = 1.0
    nmaxold = 1
    weight = 1

    def __init__(self, *args, **kwargs):
        return

    def mix(self, basemixers, nt_sG, D_asp):
        return 0.0

    def get_basemixers(self, nspins):
        return []

    def todict(self):
        return {'name': 'dummy'}


class NotMixingMixer:
    name = 'no-mixing'

    def __init__(self, beta, nmaxold, weight):
        """Construct density-mixer object.
        Parameters: they are ignored for this mixer
        """

        # whatever parameters you give it doesn't do anything with them
        self.beta = 0
        self.nmaxold = 0
        self.weight = 0

    def initialize_metric(self, gd):
        self.gd = gd
        self.metric = None

    def reset(self):
        """Reset Density-history.

        Called at initialization and after each move of the atoms.

        my_nuclei:   All nuclei in local domain.
        """

        # Previous density:
        self.nt_isG = []  # Pseudo-electron densities

    def calculate_charge_sloshing(self, R_sG):
        return self.gd.integrate(np.fabs(R_sG)).sum()

    def mix_density(self, nt_sG, D_asp, g_ss=None):
        iold = len(self.nt_isG)

        dNt = np.inf
        if iold > 0:
            # Calculate new residual (difference between input and
            # output density):
            dNt = self.calculate_charge_sloshing(nt_sG - self.nt_isG[-1])
        # Store new input density:
        self.nt_isG = [nt_sG.copy()]

        return dNt

    # may presently be overridden by passing argument in constructor
    def dotprod(self, R1_G, R2_G, dD1_ap, dD2_ap):
        pass

    def estimate_memory(self, mem, gd):
        gridbytes = gd.bytecount()
        mem.subnode('nt_iG, R_iG', 2 * self.nmaxold * gridbytes)

    def __repr__(self):
        string = 'no mixing of density'
        return string


class SeparateSpinMixerDriver:
    name = 'separate'

    def __init__(self, basemixerclass, beta, nmaxold, weight, *args, **kwargs):
        self.basemixerclass = basemixerclass

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight

    def get_basemixers(self, nspins):
        return [self.basemixerclass(self.beta, self.nmaxold, self.weight)
                for _ in range(nspins)]

    def mix(self, basemixers, nt_sG, D_asp):
        """Mix pseudo electron densities."""
        D_asp = D_asp.values()
        dNt = 0.0
        for s, (nt_G, basemixer) in enumerate(zip(nt_sG, basemixers)):
            D_a1p = [D_sp[s:s + 1] for D_sp in D_asp]
            nt_1G = nt_G[np.newaxis]
            dNt += basemixer.mix_density(nt_1G, D_a1p)
        return dNt


class SpinSumMixerDriver:
    name = 'sum'
    mix_atomic_density_matrices = False

    def __init__(self, basemixerclass, beta, nmaxold, weight):
        self.basemixerclass = basemixerclass

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight

    def get_basemixers(self, nspins):
        if nspins == 1:
            raise ValueError('Spin sum mixer expects 2 or 4 components')
        return [self.basemixerclass(self.beta, self.nmaxold, self.weight)]

    def mix(self, basemixers, nt_sG, D_asp):
        assert len(basemixers) == 1
        basemixer = basemixers[0]
        D_asp = D_asp.values()

        collinear = len(nt_sG) == 2

        # Mix density
        if collinear:
            nt_1G = nt_sG.sum(0)[np.newaxis]
        else:
            nt_1G = nt_sG[:1]

        if self.mix_atomic_density_matrices:
            if collinear:
                D_a1p = [D_sp[:1] + D_sp[1:] for D_sp in D_asp]
            else:
                D_a1p = [D_sp[:1] for D_sp in D_asp]
            dNt = basemixer.mix_density(nt_1G, D_a1p)
            if collinear:
                dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]
                for D_sp, D_1p, dD_p in zip(D_asp, D_a1p, dD_ap):
                    D_sp[0] = 0.5 * (D_1p[0] + dD_p)
                    D_sp[1] = 0.5 * (D_1p[0] - dD_p)
        else:
            dNt = basemixer.mix_density(nt_1G, D_asp)

        if collinear:
            dnt_G = nt_sG[0] - nt_sG[1]
            # Only new magnetization for spin density
            # dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]

            # Construct new spin up/down densities
            nt_sG[0] = 0.5 * (nt_1G[0] + dnt_G)
            nt_sG[1] = 0.5 * (nt_1G[0] - dnt_G)

        return dNt


class SpinSumMixerDriver2(SpinSumMixerDriver):
    name = 'sum2'
    mix_atomic_density_matrices = True


class SpinDifferenceMixerDriver:
    name = 'difference'

    def __init__(self, basemixerclass, beta, nmaxold, weight,
                 beta_m=0.7, nmaxold_m=2, weight_m=10.0):
        self.basemixerclass = basemixerclass
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        self.beta_m = beta_m
        self.nmaxold_m = nmaxold_m
        self.weight_m = weight_m

    def get_basemixers(self, nspins):
        if nspins == 1:
            raise ValueError('Spin difference mixer expects 2 or 4 components')
        basemixer = self.basemixerclass(self.beta, self.nmaxold, self.weight)
        basemixer.min_imp = 100000
        if nspins == 2:
            basemixer_m = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            basemixer_m.min_imp = 100000
            return basemixer, basemixer_m
        else:
            basemixer_x = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            basemixer_y = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            basemixer_z = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            return basemixer, basemixer_x, basemixer_y, basemixer_z

    def mix(self, basemixers, nt_sG, D_asp):
        D_asp = D_asp.values()

        if len(nt_sG) == 2:
            basemixer, basemixer_m = basemixers
        else:
            assert len(nt_sG) == 4
            basemixer, basemixer_x, basemixer_y, basemixer_z = basemixers

        if len(nt_sG) == 2:
            # Mix density
            nt_1G = nt_sG.sum(0)[np.newaxis]
            D_a1p = [D_sp[:1] + D_sp[1:] for D_sp in D_asp]
            dNt = basemixer.mix_density(nt_1G, D_a1p)

            # Mix magnetization
            dnt_1G = nt_sG[:1] - nt_sG[1:]
            dD_a1p = [D_sp[:1] - D_sp[1:] for D_sp in D_asp]
            basemixer_m.mix_density(dnt_1G, dD_a1p)
            # (The latter is not counted in dNt)

            # Construct new spin up/down densities
            nt_sG[:1] = 0.5 * (nt_1G + dnt_1G)
            nt_sG[1:] = 0.5 * (nt_1G - dnt_1G)
            for D_sp, D_1p, dD_1p in zip(D_asp, D_a1p, dD_a1p):
                D_sp[:1] = 0.5 * (D_1p + dD_1p)
                D_sp[1:] = 0.5 * (D_1p - dD_1p)
        else:
            # Mix density
            nt_1G = nt_sG[:1]
            D_a1p = [D_sp[:1] for D_sp in D_asp]
            dNt = basemixer.mix_density(nt_1G, D_a1p)

            # Mix magnetization
            Dx_a1p = [D_sp[1:2] for D_sp in D_asp]
            Dy_a1p = [D_sp[2:3] for D_sp in D_asp]
            Dz_a1p = [D_sp[3:4] for D_sp in D_asp]

            basemixer_x.mix_density(nt_sG[1:2], Dx_a1p)
            basemixer_y.mix_density(nt_sG[2:3], Dy_a1p)
            basemixer_z.mix_density(nt_sG[3:4], Dz_a1p)
        return dNt


class FullSpinMixerDriver:
    name = 'fullspin'

    def __init__(self, basemixerclass, beta, nmaxold, weight, g=None):
        self.basemixerclass = basemixerclass
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        self.g_ss = g

    def get_basemixers(self, nspins):
        if nspins == 1:
            raise ValueError('Full-spin mixer expects 2 or 4 spin channels')

        basemixer = self.basemixerclass(self.beta, self.nmaxold, self.weight)
        return [basemixer]

    def mix(self, basemixers, nt_sG, D_asp, rhot=None):
        D_asp = D_asp.values()
        basemixer = basemixers[0]
        if self.g_ss is None or len(self.g_ss) != len(nt_sG):
            self.g_ss = np.identity(len(nt_sG))

        dNt = basemixer.mix_density(nt_sG, D_asp, self.g_ss)

        return dNt


# Dictionaries to get mixers by name:
_backends = {}
_methods = {}
for cls in [FFTBaseMixer, BroydenBaseMixer, BaseMixer, NotMixingMixer, MSR1Mixer]:
    _backends[cls.name] = cls  # type:ignore
for dcls in [SeparateSpinMixerDriver, SpinSumMixerDriver,
             FullSpinMixerDriver, SpinSumMixerDriver2,
             SpinDifferenceMixerDriver, DummyMixer]:
    _methods[dcls.name] = dcls  # type:ignore


# This function is used by Density to decide mixer parameters
# that the user did not explicitly provide, i.e., it fills out
# everything that is missing and returns a mixer "driver".
def get_mixer_from_keywords(pbc, nspins, **mixerkwargs):
    if mixerkwargs.get('name') == 'dummy':
        return DummyMixer()

    if mixerkwargs.get('backend') == 'no-mixing':
        mixerkwargs['beta'] = 0
        mixerkwargs['nmaxold'] = 0
        mixerkwargs['weight'] = 0

    if nspins == 1:
        mixerkwargs['method'] = SeparateSpinMixerDriver

    # The plan is to first establish a kwargs dictionary with all the
    # defaults, then we update it with values from the user.
    kwargs = {'backend': BaseMixer}

    if np.any(pbc):  # Works on array or boolean
        kwargs.update(beta=0.08, history=16, weight=70.0)
    else:
        kwargs.update(beta=0.25, history=16, weight=1.0)

    if nspins == 1:
        kwargs['method'] = SeparateSpinMixerDriver
    else:
        kwargs['method'] = FullSpinMixerDriver

    # Clean up mixerkwargs (compatibility)
    if 'nmaxold' in mixerkwargs:
        assert 'history' not in mixerkwargs
        mixerkwargs['history'] = mixerkwargs.pop('nmaxold')

    # Now the user override:
    for key in kwargs:
        # Clean any 'None' values out as if they had never been passed:
        val = mixerkwargs.pop(key, None)
        if val is not None:
            kwargs[key] = val

    # Resolve keyword strings (like 'fft') into classes (like FFTBaseMixer):
    driver = _methods.get(kwargs['method'], kwargs['method'])
    baseclass = _backends.get(kwargs['backend'], kwargs['backend'])

    # We forward any remaining mixer kwargs to the actual mixer object.
    # Any user defined variables that do not really exist will cause an error.
    mixer = driver(baseclass, beta=kwargs['beta'],
                   nmaxold=kwargs['history'], weight=kwargs['weight'],
                   **mixerkwargs)
    return mixer


# This is the only object which will be used by Density, sod the others
class MixerWrapper:
    def __init__(self, driver, nspins, gd, world=None):
        self.driver = driver

        self.beta = driver.beta
        self.nmaxold = driver.nmaxold
        self.weight = driver.weight
        assert self.weight is not None, driver

        self.basemixers = self.driver.get_basemixers(nspins)
        for basemixer in self.basemixers:
            basemixer.initialize_metric(gd)
            basemixer.world = world

    @trace
    def mix(self, nt_sR, D_asp=None, rhot=None):
        if D_asp is not None:
            return self.driver.mix(self.basemixers, nt_sR, D_asp)

        # new interface:
        density = nt_sR
        nspins = density.nt_sR.dims[0]
        nt_sR = density.nt_sR.to_xp(np)
        D_asii = density.D_asii.to_xp(np)
        D_asp = {a: D_sii.copy().reshape((nspins, -1))
                 for a, D_sii in D_asii.items()}
        error = self.driver.mix(self.basemixers,
                                nt_sR.data,
                                D_asp)
        for a, D_sii in D_asii.items():
            ni = D_sii.shape[1]
            D_sii[:] = D_asp[a].reshape((nspins, ni, ni))
        xp = density.nt_sR.xp
        if xp is not np:
            density.nt_sR.data[:] = xp.asarray(nt_sR.data)
            density.D_asii.data[:] = xp.asarray(D_asii.data)
        return error

    def estimate_memory(self, mem, gd):
        for i, basemixer in enumerate(self.basemixers):
            basemixer.estimate_memory(mem.subnode('Mixer %d' % i), gd)

    def reset(self):
        for basemixer in self.basemixers:
            basemixer.reset()

    def __str__(self):
        lines = ['Density mixing:',
                 'Method: ' + self.driver.name,
                 'Backend: ' + self.driver.basemixerclass.name,
                 'Linear mixing parameter: %g' % self.beta,
                 f'old densities: {self.nmaxold}',
                 'Damping of long wavelength oscillations: %g' % self.weight]
        if self.weight == 1:
            lines[-1] += '  # (no daming)'
        return '\n  '.join(lines)


# Helper function to define old-style interfaces to mixers.
# Defines and returns a function which looks like a mixer class
def _definemixerfunc(method, backend):
    def getmixer(beta=None, nmaxold=None, weight=None, **kwargs):
        d = dict(method=method, backend=backend,
                 beta=beta, nmaxold=nmaxold, weight=weight)
        d.update(kwargs)
        return d
    return getmixer


Mixer = _definemixerfunc('separate', 'pulay')
MixerSum = _definemixerfunc('sum', 'pulay')
MixerSum2 = _definemixerfunc('sum2', 'pulay')
MixerDif = _definemixerfunc('difference', 'pulay')
MixerFull = _definemixerfunc('fullspin', 'pulay')
FFTMixer = _definemixerfunc('separate', 'fft')
FFTMixerSum = _definemixerfunc('sum', 'fft')
FFTMixerSum2 = _definemixerfunc('sum2', 'fft')
FFTMixerDif = _definemixerfunc('difference', 'fft')
FFTMixerFull = _definemixerfunc('fullspin', 'fft')
BroydenMixer = _definemixerfunc('separate', 'broyden')
BroydenMixerSum = _definemixerfunc('sum', 'broyden')
BroydenMixerSum2 = _definemixerfunc('sum2', 'broyden')
BroydenMixerDif = _definemixerfunc('difference', 'broyden')
