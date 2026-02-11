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
            a = 0.125 * (self.weight + 7)
            b = 0.0625 * (self.weight - 1)
            c = 0.03125 * (self.weight - 1)
            d = 0.015625 * (self.weight - 1)
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
                a = self.gd.comm.sum_scalar(
                    self.dotprod(R_1sG, mR_sG, dD_iasp[i1], dD_iasp[-1]))
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

    # may presently be overridden by passing argument in constructor
    def dotprod(self, R1_G, R2_G, dD1_ap, dD2_ap, metric=None):
        if metric is None:
            return np.vdot(R1_G, R2_G).real
        mR2_G = R2_G.copy()
        for R2, mR2 in zip(R2_G, mR2_G):
            metric(R2, mR2)
        return np.vdot(R1_G, mR2_G).real

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
    min_imp = 2.0

    def mix_density(self, nt_sG, D_asp, g_ss=None):
        nt_isG = self.nt_isG
        R_isG = self.R_isG
        D_iasp = self.D_iasp
        dD_iasp = self.dD_iasp
        spin = len(nt_sG)
        iold = len(self.nt_isG)
        dNt = np.inf
        if iold > 0:
            # Calculate new residual (difference between input and
            # output density):
            R_sG = nt_sG - nt_isG[-1]
            dNt = self.calculate_charge_sloshing(R_sG)
            R_isG.append(R_sG)
            dD_iasp.append([])
            for D_sp, D_isp in zip(D_asp, D_iasp[-1]):
                dD_iasp[-1].append(D_sp - D_isp)

            while (iold > self.nmaxold and dNt <= self.last_dNt * self.min_imp) \
                    or iold > self.nmaxold + 8:
                # Throw away too old stuff:
                del nt_isG[0]
                del R_isG[0]
                del D_iasp[0]
                del dD_iasp[0]
                # for D_p, D_ip, dD_ip in self.D_a:
                #     del D_ip[0]
                #     del dD_ip[0]
                iold = len(nt_isG)

        print(iold)
        if iold > 1:
            backtracked = False
            if dNt > self.last_dNt * self.min_imp:
                dNt = self.last_dNt
                insert_pos = 0 #-2
                tmp = nt_isG.pop()
                nt_isG.insert(insert_pos, tmp)
                tmp = R_isG.pop()
                R_isG.insert(insert_pos, tmp)
                tmp = D_iasp.pop()
                D_iasp.insert(insert_pos, tmp)
                tmp = dD_iasp.pop()
                dD_iasp.insert(insert_pos, tmp)

                R_sG = R_isG[-1]
                backtracked = True

            # 1st order norm
            ntnorm_i = np.sum(np.abs(nt_isG).reshape(iold, -1), axis=(1, )) * self.gd.dv
            self.gd.comm.sum(ntnorm_i)
            ntnorm_i = np.expand_dims(1 / ntnorm_i, axis=tuple(np.arange(1, np.array(nt_isG).ndim)))

            trust_factor = 1.2 # Account for the error introduced by the imperfections of the universe... or maybe just mixer.py

            # 2nd order norm
            # ntnorm_i = np.vecdot(np.array(nt_isG).reshape(iold, -1),
            #     np.array(nt_isG).reshape(iold, -1))
            # self.gd.comm.sum(ntnorm_i)
            # ntnorm_i = np.expand_dims(1 / np.sqrt(ntnorm_i), axis=tuple(np.arange(1, np.array(nt_isG).ndim)))

            ### 2023 Paper, Eq: 8 + 9:
            s_isG = (nt_isG[:-1] - nt_isG[-1])
            y_isG = -(R_isG[:-1] - R_isG[-1])

            # Dont
            # ts_isG = s_isG.copy()
            # ty_isG = y_isG.copy()
            ts_isG = (nt_isG[:-1] * ntnorm_i[:-1] - nt_isG[-1] * ntnorm_i[-1])
            ty_isG = -(R_isG[:-1] * ntnorm_i[:-1] - R_isG[-1] * ntnorm_i[-1])
            for ty_sG, ts_sG in zip(ty_isG, ts_isG):
                for ty_G, ts_G in zip(ty_sG, ts_sG):
                    if self.metric is not None:
                        self.metric(ty_G, ty_G)
                        self.metric(ts_G, ts_G)
                if g_ss is not None:
                    ty_sG[:] = np.tensordot(g_ss, ty_sG, axes=(1, 0))
                    ts_sG[:] = np.tensordot(g_ss, ts_sG, axes=(1, 0))

            ### 2023 paper eq 22 - Limit good_broydenness
            YY_LIM = y_isG.reshape((iold - 1, -1)) @ ty_isG.reshape((iold - 1, -1)).T
            self.gd.comm.sum(YY_LIM)
            YY_LIM = np.linalg.norm(YY_LIM, ord='fro')
            YS_LIM = y_isG.reshape((iold - 1, -1)) @ ts_isG.reshape((iold - 1, -1)).T
            self.gd.comm.sum(YS_LIM)
            YS_LIM = np.linalg.norm(YS_LIM, ord='fro')
            max_gb = np.clip(YY_LIM / YS_LIM, 1, 25)  # Take care
            good_broydenness = 0.5 * max_gb

            # Choose max good_broydenness s.t. A_ii is positive definite
            # for good_broydenness in good_broydenness_range:
            # binary search 2**(-8) accuracy:
            t_norm = np.vecdot(ty_isG.reshape((iold - 1, -1)), y_isG.reshape((iold - 1, -1)))
            self.gd.comm.sum(t_norm)
            t_norm = np.sqrt(t_norm)
            for iter in range(2, 9):
                t_isG = (ty_isG + good_broydenness * ts_isG).reshape(
                    (iold - 1, -1)) / t_norm[:, None]

                A_ii = t_isG @ (y_isG.reshape((iold - 1, -1)) / t_norm[:, None]).T
                self.gd.comm.sum(A_ii)
                try:
                    eigs = np.linalg.eigvals(A_ii)
                except np.linalg.LinAlgError:
                    good_broydenness -= 2**(-iter) * max_gb
                    continue
                if np.all(eigs.real > 0) and np.all(eigs.imag == 0):
                    good_broydenness += 2**(-iter) * max_gb
                else:
                    good_broydenness -= 2**(-iter) * max_gb
            good_broydenness -= 2**(-iter) * max_gb

            # Don't increase good-broydeness too quickly:
            # good_broydenness = min(
            #     good_broydenness,
            #     max(1, self.last_good_broydeness) * (1.3 if not backtracked else 1))
            # self.last_good_broydeness = good_broydenness
            # good_broydenness *= 1 / trust_factor

            # Do not good broyden when density is crap
            crabiness_mult = 3e-2 / (dNt * ntnorm_i.ravel()[-1])
            # print('crab_factor: ', min(0.9, crabiness_mult))
            good_broydenness *= min(1 / trust_factor, crabiness_mult)
            print('good_broydenness: ', good_broydenness)
            t_isG = ty_isG + good_broydenness * ts_isG
            A_ii = t_isG.reshape((iold - 1, -1)) @ y_isG.reshape((iold - 1, -1)).T
            self.gd.comm.sum(A_ii)
            t_norm = 1 / np.sqrt(np.diag(A_ii))

            ### Scale the problem!
            s_isG *= np.expand_dims(t_norm, axis=tuple(np.arange(1, s_isG.ndim)))
            y_isG *= np.expand_dims(t_norm, axis=tuple(np.arange(1, y_isG.ndim)))

            sD_iasp = []
            yD_iasp = []
            for i1 in range(len(D_iasp) - 1):
                sD_asp = []
                yD_asp = []
                for a1 in range(len(D_iasp[i1])):
                    sD_asp.append((D_iasp[i1][a1] - D_iasp[-1][a1]) * t_norm[i1])
                    yD_asp.append(-(dD_iasp[i1][a1] - dD_iasp[-1][a1]) * t_norm[i1])
                sD_iasp.append(sD_asp)
                yD_iasp.append(yD_asp)
            ###

            # Normalize t
            ts_isG *= np.expand_dims(t_norm, axis=tuple(np.arange(1, s_isG.ndim)))
            ty_isG *= np.expand_dims(t_norm, axis=tuple(np.arange(1, y_isG.ndim)))
            t_isG = ty_isG + good_broydenness * ts_isG  # Also known as W depending on the paper

            A_ii = t_isG.reshape((iold - 1, -1)) @ y_isG.reshape((iold - 1, -1)).T
            self.gd.comm.sum(A_ii)

            B_ii = t_isG.reshape((iold - 1, -1)) @ s_isG.reshape((iold - 1, -1)).T
            self.gd.comm.sum(B_ii)

            # This parameter is surprisingly important for stability
            # 2e-4 seems to work well for most systems
            weight = 1e-4

            ### SVD Regularization:
            S, V, D = np.linalg.svd(A_ii)
            V = V / (V**2 + (weight * np.max(V))**2)
            A_ii = D.T @ np.diag(V) @ S.T
            S, V, D = np.linalg.svd(B_ii)
            V = V / (V**2 + (weight * np.max(V))**2)
            B_ii = D.T @ np.diag(V) @ S.T

            ### Moore-Penrose:
            # A_ii = np.linalg.solve(
            #     A_ii.T @ A_ii + (normA * alphaA * np.eye(A_ii.shape[0]))**2, A_ii)
            # B_ii = np.linalg.solve(
            #     B_ii.T @ B_ii + (normB * alphaB * np.eye(B_ii.shape[0]))**2, B_ii)

            ### Rawdog Inverse:
            # A_ii = np.linalg.inv(A_ii)
            # B_ii = np.linalg.inv(B_ii)

            # H_isG = (A_ii @ t_isG.reshape((iold - 1, -1))).reshape(t_isG.shape)
            # B_isG = (B_ii @ t_isG.reshape((iold - 1, -1))).reshape(t_isG.shape)

            # 2023 paper eq 14 alpha_i = Inv(Y_n^T @ W) @ W^T @ Res_n part
            # A_ii should not be transposed... Or should be... Depending on
            # what paper your read, transposed works best, but most papers
            # say not to, so... rip
            alpha_i = t_isG.reshape((iold - 1, -1)) @ R_sG.reshape((-1))
            self.gd.comm.sum(alpha_i)
            alpha_i = A_ii @ alpha_i
            if self.world:
                self.world.broadcast(alpha_i, 0)

            ######### Stuff for predicting mixing coefficients:
            A1 = self.uk_sG.reshape(-1) @ self.uk_sG.reshape(-1)
            A1 = self.gd.comm.sum_scalar(A1)
            B1 = (self.R_isG[-2] - self.uk_sG).reshape(-1) @ \
                (self.R_isG[-2] - self.uk_sG).reshape(-1)
            B1 = self.gd.comm.sum_scalar(B1)

            A2_i = t_isG.reshape((iold - 1, -1)) @ self.uk_sG.reshape(-1)
            self.gd.comm.sum(A2_i)
            B2_i = t_isG.reshape((iold - 1, -1)) @ self.pk_sG.reshape(-1)
            self.gd.comm.sum(B2_i)
            A3_i = y_isG.reshape((iold - 1, -1)) @ self.uk_sG.reshape(-1)
            self.gd.comm.sum(A3_i)
            B3_i = y_isG.reshape((iold - 1, -1)) @ (self.R_isG[-2] - self.uk_sG).reshape(-1)
            self.gd.comm.sum(B3_i)

            A2 = A3_i @ B_ii @ A2_i * trust_factor
            B2 = B3_i @ B_ii @ B2_i

            if iold != 2:
                B0_ratio = (
                    self.B0 + np.clip(np.abs(B1 / B2), 0.3, 1)
                    ) / (2 * self.B0)
                self.B0 *= np.clip(B0_ratio, 0.67,
                   1.5 if not backtracked else 1.0)
            else:
                self.B0 = 1

            A0_ratio = (self.A0 + np.clip(
                np.abs(A1 / A2),
                0.035,
                self.beta + (max(self.beta, 1) - self.beta) #  * min(1, (iold + 1) / self.nmaxold)
                )
            ) / (2 * self.A0)
            self.A0 *= np.clip(A0_ratio, 0.67, 1.5 if not backtracked else 1.0)

            A0 = self.A0
            B0 = self.B0
            if self.gd.comm.rank == 0:
                print(f"rank: {self.world.rank}, A0: {A0}, B0: {B0}")

            trust_radius = 1 * np.sum((A0 * self.uk_sG + B0 * self.pk_sG)**2)
            if self.trust_radius is not None:
                trust_radius = (self.trust_radius + self.gd.comm.sum_scalar(trust_radius)**0.5) / 2
                if backtracked:
                    self.trust_radius = min(self.trust_radius, trust_radius)
                else:
                    self.trust_radius = trust_radius
            else:
                self.trust_radius = self.gd.comm.sum_scalar(trust_radius)**0.5

            self.uk_sG = np.zeros_like(nt_sG)
            self.pk_sG = np.zeros_like(nt_sG)

            for i1, alpha in enumerate(alpha_i):
                self.uk_sG -= y_isG[i1] * alpha
                self.pk_sG += s_isG[i1] * alpha

            self.uk_sG += R_sG
            step_sG = A0 * self.uk_sG + B0 * self.pk_sG
            step_size = np.sum((B0 * self.pk_sG)**2)
            step_size = self.gd.comm.sum_scalar(step_size)**0.5

            beta_i = alpha_i.copy()
            scale_factor = 1

            if step_size >= self.trust_radius * 1.01:
                # Time to mix the mixing...
                # print('XXXX: Performing trust region control!!!')
                # print(f'XXXX {step_size} > {self.trust_radius}')
                ### Perform lsq squares with lagrange multiplier
                # B^T R:
                BR_i = t_isG.reshape((iold - 1, -1)) @ R_sG.reshape(-1)
                self.gd.comm.sum(BR_i)

                # f^T f
                s_ii = B0**2 * s_isG.reshape((iold - 1, -1)) @ s_isG.reshape((iold - 1, -1)).T
                self.gd.comm.sum(s_ii)

                A2_ii = np.linalg.inv(A_ii)

                # Optimize (ridge regression):
                def err_fct(lamb):
                    beta_i = np.linalg.solve(
                        A2_ii + np.exp(lamb) * np.eye(iold - 1), BR_i
                    )
                    return beta_i @ s_ii @ beta_i - self.trust_radius**2

                from scipy.optimize import root_scalar
                lamb = root_scalar(err_fct, bracket=[-15, 15])
                beta_i = np.linalg.solve(
                    A2_ii + np.exp(lamb.root) * np.eye(iold - 1), BR_i
                )
                if self.world:
                    self.world.broadcast(beta_i, 0)

                self.uk_sG[:] = 0
                self.pk_sG[:] = 0
                alpha_i[:] = beta_i

                for i1, (alpha, beta) in enumerate(zip(alpha_i, beta_i)):
                    self.uk_sG -= y_isG[i1] * alpha
                    self.pk_sG += s_isG[i1] * beta

                new_step_size = np.sum((B0 * self.pk_sG)**2)
                new_step_size = self.gd.comm.sum_scalar(new_step_size)**0.5
                scale_factor = new_step_size / step_size

                self.uk_sG += R_sG
                self.uk_sG *= scale_factor
                step_sG = A0 * self.uk_sG + B0 * self.pk_sG

            nt_sG[:] = nt_isG[-1] + step_sG

            for a1, D_sp in enumerate(D_asp):
                D_sp[:] = D_iasp[-1][a1] + A0 * dD_iasp[-1][a1] * scale_factor

            for i1, (alpha, beta) in enumerate(zip(alpha_i, beta_i)):
                for a1, D_sp in enumerate(D_asp):
                    D_sp -= A0 * alpha * yD_iasp[i1][a1] * scale_factor
                    D_sp += B0 * beta * sD_iasp[i1][a1]

            # Sync the density, because apparantly they cant agree...
            if self.world:
                nt_sR = self.gd.collect(nt_sG, broadcast=True)
                self.world.broadcast(nt_sR, 0)
                nt_sG[:] = self.gd.distribute(nt_sR)

        elif iold == 1:
            # Pratt step
            self.trust_radius = None
            self.last_good_broydeness = 5
            self.A0 = self.beta
            A0 = self.beta * 0.67
            self.uk_sG = R_sG
            self.pk_sG = np.zeros_like(self.uk_sG)
            nt_sG[:] = nt_isG[-1] + A0 * self.uk_sG
            for a1, D_sp in enumerate(D_asp):
                D_sp[:] = D_iasp[-1][a1] + A0 * dD_iasp[-1][a1]

        # Store new input density (and new atomic density matrices):
        nt_isG.append(nt_sG.copy())
        D_iasp.append([])
        for D_sp in D_asp:
            D_iasp[-1].append(D_sp.copy())
        self.last_dNt = dNt
        return dNt


class ExperimentalDotProd:
    def __init__(self, calc):
        self.calc = calc

    def __call__(self, R1_G, R2_G, dD1_ap, dD2_ap):
        prod = np.vdot(R1_G, R2_G).real
        setups = self.calc.wfs.setups
        # okay, this is a bit nasty because it depends on dD1_ap
        # and its friend having come from D_asp.values() and the dictionaries
        # not having been modified.  This is probably true... for now.
        avalues = self.calc.density.D_asp.keys()
        for a, dD1_p, dD2_p in zip(avalues, dD1_ap, dD2_ap):
            I4_pp = setups[a].four_phi_integrals()
            dD4_pp = np.outer(dD1_p, dD2_p)  # not sure if corresponds quite
            prod += (I4_pp * dD4_pp).sum()
        return prod


class ReciprocalMetric:
    def __init__(self, weight, k2_Q):
        self.k2_Q = k2_Q
        k2_min = np.min(self.k2_Q)
        self.q1 = (weight - 1) * k2_min

    def __call__(self, R_Q, mR_Q):
        mR_Q[:] = R_Q * (1.0 + self.q1 / self.k2_Q)


class FFTBaseMixer(BaseMixer):
    name = 'fft'

    """Mix the density in Fourier space"""
    def __init__(self, beta, nmaxold, weight):
        super().__init__(beta, nmaxold, weight)
        self.gd1 = None

    def initialize_metric(self, gd):
        self.gd = gd

        if gd.comm.rank == 0:
            self.gd1 = gd.new_descriptor(comm=mpi.serial_comm)
            k2_Q, _ = construct_reciprocal(self.gd1)
            self.metric = ReciprocalMetric(self.weight, k2_Q)
        else:
            self.metric = lambda R_Q, mR_Q: None

    def calculate_charge_sloshing(self, R_sQ):
        if self.gd.comm.rank == 0:
            assert R_sQ.ndim == 4  # and len(R_sQ) == 1
            cs = sum(self.gd1.integrate(np.fabs(ifftn(R_Q).real))
                     for R_Q in R_sQ)
        else:
            cs = 0.0
        return self.gd.comm.sum_scalar(cs)

    def mix_density(self, nt_sR, D_asp, g_ss=None, rhot=None):
        # Transform real-space density to Fourier space
        nt1_sR = [self.gd.collect(nt_R) for nt_R in nt_sR]
        if self.gd.comm.rank == 0:
            nt1_sG = np.ascontiguousarray([fftn(nt_R, norm='backward') for nt_R in nt1_sR])
        else:
            nt1_sG = np.empty((len(nt_sR), 0, 0, 0), dtype=complex)

        dNt = super().mix_density(nt1_sG, D_asp)

        # Return density in real space
        for nt_G, nt_R in zip(nt1_sG, nt_sR):
            if self.gd.comm.rank == 0:
                nt1_R = ifftn(nt_G, norm='backward').real
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
