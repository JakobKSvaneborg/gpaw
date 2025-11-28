import numpy as np


class LBFGS:
    def __init__(self, kpt_comm, array_shape, dtype, memory=5):
        """
        L-BFGS optimizer initialization.

        Parameters
        ----------
        array_shape : tuple
            Shape of the parameter array.
        dtype : type
            Data type of arrays (float or complex).
        kpt_comm : object
            Communication object for summing quantities over k-points
        memory : int
            Number of past steps to store for the L-BFGS approximation.
        """

        # Counter for number of local iterations performed
        self.local_iter = 0

        # Maximum number of previous steps to store for limited-memory Hessian
        self.memory = memory

        # Previous gradient and variable arrays
        self.g_old = None
        self.a_old = None

        # Current search direction
        self.search_dir = None

        # Arrays to store last 'memory' differences in variables and gradients
        # ds[m] ~ change in parameters (search direction)
        # dy[m] ~ change in gradients
        self.ds = np.zeros(shape=(memory,) + array_shape, dtype=dtype)
        self.dy = np.zeros(shape=(memory,) + array_shape, dtype=dtype)

        # Scaling factors for L-BFGS (1 / (y^T * s))
        self.rho = np.zeros(memory, dtype=dtype)

        # Communication object for parallel sum across k-points
        self.kpt_comm = kpt_comm

    def update(self, a_cur, g_cur):
        """
        Compute the next search direction using L-BFGS two-loop recursion.

        Parameters
        ----------
        a_cur : ndarray
            Current variables (parameters).
        g_cur : ndarray
            Current gradient of objective function.

        Returns
        -------
        search_dir : ndarray
            Updated search direction for the next iteration.
        """

        # Step 1: First iteration initialization
        if self.local_iter == 0:
            # Store current gradient and variables
            self.g_old = g_cur.copy()
            self.a_old = a_cur.copy()

            # Initial search direction is simply the negative gradient
            self.search_dir = -g_cur

            # Increment iteration counter
            self.local_iter += 1

            # Return initial search direction
            return self.search_dir

        else:
            # Step 2: Determine memory index for circular storage
            m = self.local_iter % self.memory

            # Store changes in variables (ds) and gradients (dy)
            # For first few iterations, ds = previous search direction
            self.ds[m] = self.search_dir.copy()
            self.dy[m] = g_cur - self.g_old

            # Compute curvature y^T * s
            dyds = np.sum(self.dy[m].conj() * self.ds[m]).real
            # Sum over k-points for parallel calculations
            dyds = self.kpt_comm.sum_scalar(dyds)

            # Compute L-BFGS scaling factor rho = 1 / (y^T * s)
            if abs(dyds) > 1.0e-20:
                self.rho[m] = 1.0 / dyds
            else:
                # Avoid division by zero
                self.rho[m] = 1.0e20

            # Step 3: Stability check
            if self.rho[m] < 0:
                self.local_iter = 0
                self.rho *= 0
                self.ds *= 0
                self.dy *= 0
                return self.update(a_cur, g_cur)

            # Step 4: Two-loop recursion to compute search direction
            q = g_cur.copy()

            # First loop: backward over stored vectors
            k = self.memory - 1
            alpha = np.zeros_like(self.rho)

            while k > -1:
                # Circular indexing over memory
                c_ind = (k + m + 1) % self.memory
                k -= 1

                # Dot product s^T * q
                sq = np.sum(self.ds[c_ind].conj() * q).real
                sq = self.kpt_comm.sum_scalar(sq)

                # Scaling factor for this step
                alpha[c_ind] = self.rho[c_ind] * sq

                # Update q
                q -= alpha[c_ind] * self.dy[c_ind]

            # Step 5: Scale by initial Hessian approximation
            yy = np.sum(self.dy[m].conj() * self.dy[m]).real
            yy = self.kpt_comm.sum_scalar(yy)
            # avoid divide by zero
            devis = np.maximum(self.rho[m] * yy, 1.0e-20)
            self.search_dir = q / devis

            # Second loop: forward over stored vectors
            for k in range(self.memory):
                if self.local_iter < self.memory:
                    c_ind = k
                else:
                    c_ind = (k + m + 1) % self.memory

                # Compute beta = rho * y^T * r
                yr = np.sum(self.dy[c_ind].conj() * self.search_dir).real
                yr = self.kpt_comm.sum_scalar(yr)
                beta = self.rho[c_ind] * yr

                # Update search direction with alpha and beta corrections
                self.search_dir += self.ds[c_ind] * (alpha[c_ind] - beta)

            # Step 6: Update stored previous gradient and variable
            self.g_old = g_cur.copy()
            self.a_old = a_cur.copy()

            # Step 7: get descent direction
            self.search_dir *= -1

            self.local_iter += 1

            # Return updated search direction
            return self.search_dir


class LBFGSAdapter:
    """
    Adapter to allow the new NumPy-based L-BFGS
    to replace the Vector-based LBFGS.
    """

    def __init__(self, array_shape, kpt_comm, dtype, memory=5):
        self.lbfgs_new = LBFGS(kpt_comm, array_shape, dtype, memory)

    def update(self, psit_unX, pg_unX):
        """
        Convert old-style vectors to NumPy arrays,
        call new L-BFGS, and convert back.
        """
        # Convert old vectors to NumPy arrays
        a_cur = np.stack([x.data for x in psit_unX])
        g_cur = np.stack([g.data for g in pg_unX])

        # Call the new LBFGS
        p_cur = self.lbfgs_new.update(a_cur, g_cur)

        # Convert back to old-style objects
        p_unX_new = []
        for p_vec, old_vec in zip(p_cur, psit_unX):
            new_vec = old_vec.new()  # create empty vector same type as old
            new_vec.data[:] = p_vec
            p_unX_new.append(new_vec)

        return p_unX_new
