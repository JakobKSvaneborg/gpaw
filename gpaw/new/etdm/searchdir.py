import numpy as np


class LBFGS:
    def __init__(self, array_shape, kpt_comm, dtype, memory=5):

        self.local_iter = 0
        self.memory = memory

        self.g_old = None
        self.a_old = None
        self.search_dir = None

        self.ds = np.zeros(shape=(memory,) + array_shape, dtype=dtype)
        self.dy = np.zeros(shape=(memory,) + array_shape, dtype=dtype)
        self.rho = np.zeros(memory, dtype=dtype)

        self.kpt_comm = kpt_comm

    def update(self, a_cur, g_cur):

        if self.local_iter == 0:
            self.g_old = g_cur.copy()
            self.a_old = a_cur.copy()
            self.search_dir = -g_cur
            self.local_iter += 1
            return self.search_dir
        else:
            m = self.local_iter % self.memory
            # change in ds = a_cur - a_old, which is search dir
            # but for some algos only the difference is known
            self.ds[m] = self.search_dir.copy()
            self.dy[m] = g_cur - self.g_old
            dyds = np.sum(self.dy[m].conj() * self.ds[m]).real
            dyds = self.kpt_comm.sum_scalar(dyds)

            if abs(dyds) > 1.0e-20:
                self.rho[m] = 1.0 / dyds
            else:
                self.rho[m] = 1.0e20

            if self.rho[m] < 0:
                self.local_iter = 0
                self.rho *= 0
                self.ds *= 0
                self.dy *= 0
                return self.update(a_cur, g_cur)

            q = g_cur.copy()

            k = self.memory - 1
            alpha = np.zeros_like(self.rho)
            while k > -1:
                c_ind = (k + m + 1) % self.memory
                k -= 1

                sq = np.sum(self.ds[c_ind].conj() * q).real
                sq = self.kpt_comm.sum_scalar(sq)

                alpha[c_ind] = self.rho[c_ind] * sq
                q -= alpha[c_ind] * self.dy[c_ind]

            yy = np.sum(self.dy[m].conj() * self.dy[m]).real
            yy = self.kpt_comm.sum_scalar(yy)
            devis = np.maximum(self.rho[m] * yy, 1.0e-20)
            self.search_dir = q / devis

            for k in range(self.memory):
                if self.local_iter < self.memory:
                    c_ind = k
                else:
                    c_ind = (k + m + 1) % self.memory

                yr = np.sum(self.dy[c_ind].conj() * self.search_dir).real
                yr = self.kpt_comm.sum_scalar(yr)

                beta = self.rho[c_ind] * yr
                self.search_dir += self.ds[c_ind] * (alpha[c_ind] - beta)

            self.g_old = g_cur.copy()
            self.a_old = a_cur.copy()
            self.search_dir *= -1
            self.local_iter += 1

            return self.search_dir
