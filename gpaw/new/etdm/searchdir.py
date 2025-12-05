import numpy as np


class MultiXArrays:
    def __init__(self, a_unX, weihgts, comm):
        self.a_unX = a_unX
        self.weights = weihgts
        self.comm = comm

    def new(self, a_unX):
        return MultiXArrays(a_unX, self.weights, self.comm)

    def copy(self):
        return self.new(
            [a_nX.copy() for a_nX in self.a_unX])

    def __neg__(self):
        b_unX = self.copy()
        for b_nX in b_unX.a_unX:
            b_nX.data *= -1.0
        return b_unX

    def __iadd__(self, other):
        for a_nX, b_nX in zip(self.a_unX, other.a_unX):
            a_nX.data += b_nX.data
        return self

    def __sub__(self, other):
        a_unX = self.copy()
        for a_nX, b_nX in zip(a_unX.a_unX, other.a_unX):
            a_nX.data -= b_nX.data
        return a_unX

    def __mul__(self, other):
        a_unX = self.copy()
        for a_nX in a_unX.a_unX:
            a_nX.data *= other
        return a_unX

    __rmul__ = __mul__

    def __matmul__(self, other):
        return self.comm.sum_scalar(
            sum(weight * a_nX.trace_inner_product(b_nX)
                for weight, a_nX, b_nX
                in zip(self.weights, self.a_unX, other.a_unX)))


class LBFGS:
    def __init__(self,
                 *,
                 memory=5):
        """
        L-BFGS optimizer initialization.

        Parameters
        ----------
        array_shape : tuple
            Shape of the parameter array.
        dtype : type
            Data type of arrays (float or complex).
        array_unX : list[UGArray]
            Distributed parameter array.
        kpt_comm : object
            Communication object for summing quantities over k-points
        memory : int
            Number of past steps to store for the L-BFGS approximation.
        """

        # Maximum number of previous steps to store for limited-memory Hessian
        self.memory = memory

        # init empty arrays
        self.reset()

    def reset(self):
        # Counter for number of local iterations performed
        self.local_iter = 0

        # Current search direction
        self.search_dir = None

        # Previous gradient and variable arrays
        self.g_old = None
        self.a_old = None

        # Arrays to store last 'memory' differences in variables and gradients
        # ds[m] ~ change in parameters (search direction)
        # dy[m] ~ change in gradients
        self.ds = ['None'] * self.memory
        self.dy = ['None'] * self.memory

        # Scaling factors for L-BFGS (1 / (y^T * s))
        self.rho = np.zeros(self.memory)

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
            self.ds = [0.0 * g_cur for _ in range(self.memory)]
            self.dy = [0.0 * g_cur for _ in range(self.memory)]

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
            dyds = self.dy[m] @ self.ds[m]

            # Compute L-BFGS scaling factor rho = 1 / (y^T * s)
            if abs(dyds) > 1.0e-20:
                self.rho[m] = 1.0 / dyds
            else:
                # Avoid division by zero
                self.rho[m] = 1.0e20

            # Step 3: Stability check
            if self.rho[m] < 0:
                self.local_iter = 0
                self.rho[:] = 0.0
                self.ds[:] = 0.0
                self.dy[:] = 0.0
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
                sq = self.ds[c_ind] @ q

                # Scaling factor for this step
                alpha[c_ind] = self.rho[c_ind] * sq

                # Update q
                q += -alpha[c_ind] * self.dy[c_ind]

            # Step 5: Scale by initial Hessian approximation
            yy = self.dy[m] @ self.dy[m]
            # avoid divide by zero
            devis = np.maximum(self.rho[m] * yy, 1.0e-20)
            self.search_dir = q * (1 / devis)

            # Second loop: forward over stored vectors
            for k in range(self.memory):
                if self.local_iter < self.memory:
                    c_ind = k
                else:
                    c_ind = (k + m + 1) % self.memory

                # Compute beta = rho * y^T * r
                yr = self.dy[c_ind] @ self.search_dir
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
